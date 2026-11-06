import json

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src import cache_metadata, experiment, features, model, scaling


def training_tables(n=40):
    rng = np.random.default_rng(7)
    raw = pd.DataFrame(rng.random((n, len(features.FEATURE_NAMES))),
                       columns=features.FEATURE_NAMES,
                       index=[f"p{i}" for i in range(n)])
    scaler = scaling.fit_scaler(raw)
    scaled = scaling.apply_scaler(scaler, raw)
    return scaled, scaled[model.TARGET], scaler


def model_cache(tmp_path, name="toy", seed=11):
    x, y, scaler = training_tables()
    columns = model.FEATURES_WITHOUT_PAGERANK
    forest = model.load_or_train(name, x[columns], y, tmp_path, seed=seed, scaler=scaler)
    path = tmp_path / f"rf_{name}.joblib"
    return x, y, scaler, forest, path


def experiment_inputs():
    graph = nx.DiGraph([("3", "2"), ("3", "1"), ("2", "1"), ("4", "2")])
    raw = features.compute_features(graph)
    scaler = scaling.fit_scaler(raw)
    scaled = scaling.apply_scaler(scaler, raw)
    forests = {
        name: model.train_model(scaled[columns], scaled[model.TARGET], seed=13,
                                cache_name=name, scaler=scaler)
        for name, columns in experiment.VARIANTS.items()
    }
    return graph, scaler, forests


def run_small_experiment(tmp_path, graph=None, scaler=None, forests=None, k=1):
    if graph is None:
        graph, scaler, forests = experiment_inputs()
    experiment.run_all(graph, scaler, forests, k=k, kinds=("remove",), levels=(0.0,),
                       seed=17, runs_dir=tmp_path)
    return tmp_path / "remove_00.npz"


def test_matching_model_metadata_reuses_cache(tmp_path, monkeypatch):
    x, y, scaler, first, path = model_cache(tmp_path)
    before = path.read_bytes()

    def must_not_train(*args, **kwargs):
        raise AssertionError("valid model cache was not reused")

    monkeypatch.setattr(model, "train_model", must_not_train)
    second = model.load_or_train("toy", x[model.FEATURES_WITHOUT_PAGERANK], y,
                                 tmp_path, seed=11, scaler=scaler)
    assert path.read_bytes() == before
    assert model.model_fingerprint(second) == model.model_fingerprint(first)


@pytest.mark.parametrize("change", ["seed", "data", "feature_definition", "model_settings"])
def test_model_metadata_changes_invalidate_without_overwriting(tmp_path, monkeypatch, change):
    x, y, scaler, _, path = model_cache(tmp_path)
    metadata = cache_metadata.metadata_path(path)
    artifact_before, metadata_before = path.read_bytes(), metadata.read_bytes()
    kwargs = {"seed": 11, "scaler": scaler}
    changed_x = x[model.FEATURES_WITHOUT_PAGERANK].copy()

    if change == "seed":
        kwargs["seed"] = 12
    elif change == "data":
        changed_x.iloc[0, 0] += 0.01
    elif change == "feature_definition":
        monkeypatch.setitem(features.FEATURE_DEFINITIONS, "degree", "changed definition")
    elif change == "model_settings":
        monkeypatch.setitem(model.FOREST_SETTINGS, "n_estimators", 201)

    with pytest.raises(cache_metadata.CacheValidationError):
        model.load_or_train("toy", changed_x, y, tmp_path, **kwargs)
    assert path.read_bytes() == artifact_before
    assert metadata.read_bytes() == metadata_before


@pytest.mark.parametrize("mode", ["missing", "corrupt"])
def test_missing_or_corrupt_model_metadata_is_rejected_safely(tmp_path, mode):
    x, y, scaler, _, path = model_cache(tmp_path)
    metadata = cache_metadata.metadata_path(path)
    artifact_before = path.read_bytes()
    if mode == "missing":
        metadata.unlink()
    else:
        metadata.write_text("{not-json", encoding="utf-8")

    with pytest.raises(cache_metadata.CacheValidationError):
        model.load_or_train("toy", x[model.FEATURES_WITHOUT_PAGERANK], y,
                            tmp_path, seed=11, scaler=scaler)
    assert path.read_bytes() == artifact_before


def test_matching_experiment_metadata_reuses_cache(tmp_path, monkeypatch):
    graph, scaler, forests = experiment_inputs()
    path = run_small_experiment(tmp_path, graph, scaler, forests)
    before = path.read_bytes()

    def must_not_run(*args, **kwargs):
        raise AssertionError("valid experiment cache was not reused")

    monkeypatch.setattr(experiment, "run_setting", must_not_run)
    experiment.run_all(graph, scaler, forests, k=1, kinds=("remove",), levels=(0.0,),
                       seed=17, runs_dir=tmp_path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["k", "seed", "graph", "scaler", "model"])
def test_experiment_metadata_changes_invalidate_without_overwriting(tmp_path, change):
    graph, scaler, forests = experiment_inputs()
    path = run_small_experiment(tmp_path, graph, scaler, forests)
    metadata = cache_metadata.metadata_path(path)
    artifact_before, metadata_before = path.read_bytes(), metadata.read_bytes()
    k, seed = 1, 17

    if change == "k":
        k = 2
    elif change == "seed":
        seed = 18
    elif change == "graph":
        graph = graph.copy()
        graph.add_edge("5", "1")
    elif change == "scaler":
        altered = features.compute_features(graph).copy()
        altered.iloc[0, 0] *= 2
        scaler = scaling.fit_scaler(altered)
    elif change == "model":
        forests["without_pagerank"]._reproducibility_fingerprint = "different-model"

    with pytest.raises(cache_metadata.CacheValidationError):
        experiment.run_all(graph, scaler, forests, k=k, kinds=("remove",), levels=(0.0,),
                           seed=seed, runs_dir=tmp_path)
    assert path.read_bytes() == artifact_before
    assert metadata.read_bytes() == metadata_before


def test_model_and_experiment_caches_are_validated_independently(tmp_path):
    models_dir, runs_dir = tmp_path / "models", tmp_path / "runs"
    x, y, scaler = training_tables()
    forests = {
        name: model.load_or_train(name, x[columns], y, models_dir, seed=19, scaler=scaler)
        for name, columns in experiment.VARIANTS.items()
    }
    graph = nx.DiGraph([("3", "2"), ("2", "1")])
    experiment_path = run_small_experiment(runs_dir, graph, scaler, forests)
    experiment_metadata = cache_metadata.metadata_path(experiment_path)
    experiment_metadata_before = experiment_metadata.read_bytes()

    model_path = models_dir / "rf_with_pagerank.joblib"
    model_metadata = cache_metadata.metadata_path(model_path)
    model_metadata_before = model_metadata.read_bytes()

    model_metadata.write_text("broken", encoding="utf-8")
    scores, nodes, methods = experiment.load_setting("remove", 0.0, runs_dir=runs_dir)
    assert scores.shape[0] == 1 and len(nodes) == len(graph) and methods == experiment.METHODS

    model_metadata.write_bytes(model_metadata_before)
    experiment_metadata.write_text(json.dumps({"schema_version": -1}), encoding="utf-8")
    reused = model.load_or_train("with_pagerank", x[model.FEATURES_WITH_PAGERANK], y,
                                 models_dir, seed=19, scaler=scaler)
    assert model.model_fingerprint(reused) == model.model_fingerprint(forests["with_pagerank"])
    assert experiment_path.exists()
    experiment_metadata.write_bytes(experiment_metadata_before)


def test_corrupt_experiment_metadata_is_rejected(tmp_path):
    path = run_small_experiment(tmp_path)
    before = path.read_bytes()
    cache_metadata.metadata_path(path).write_text("not json", encoding="utf-8")
    with pytest.raises(cache_metadata.CacheValidationError):
        experiment.load_setting("remove", 0.0, runs_dir=tmp_path)
    assert path.read_bytes() == before
