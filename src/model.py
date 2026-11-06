"""Block C - Random Forest that maps the feature vector of a node to an importance score."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from . import cache_metadata, config, features

TARGET = "pagerank"

# the two versions of the model (decision D2 in docs/decisions.md)
FEATURES_WITH_PAGERANK = ["degree", "betweenness", "closeness", "pagerank", "kcore"]
FEATURES_WITHOUT_PAGERANK = ["degree", "betweenness", "closeness", "kcore"]

FOREST_SETTINGS = dict(n_estimators=200, min_samples_leaf=5, n_jobs=-1)


def _model_spec(name, x, y, seed, scaler=None):
    estimator = RandomForestRegressor(random_state=seed, **FOREST_SETTINGS)
    return {
        "artifact_type": "random_forest",
        "name": name,
        "dataset_fingerprint": cache_metadata.stable_fingerprint({
            "features": cache_metadata.table_fingerprint(x),
            "target": cache_metadata.table_fingerprint(y),
        }),
        "features": {
            "names": list(x.columns),
            "definitions": {column: features.FEATURE_DEFINITIONS[column] for column in x.columns},
            "target": TARGET,
        },
        "model": {
            "class": f"{type(estimator).__module__}.{type(estimator).__name__}",
            "hyperparameters": estimator.get_params(deep=False),
        },
        "training_seed": seed,
        "dependencies": cache_metadata.dependency_versions(),
        "scaler": cache_metadata.scaler_spec(scaler),
    }


def model_fingerprint(forest):
    fingerprint = getattr(forest, "_reproducibility_fingerprint", None)
    if fingerprint is not None:
        return fingerprint
    runtime = {
        "class": f"{type(forest).__module__}.{type(forest).__name__}",
        "hyperparameters": forest.get_params(deep=False),
        "feature_names": list(getattr(forest, "feature_names_in_", [])),
        "dependencies": cache_metadata.dependency_versions(),
    }
    return cache_metadata.stable_fingerprint(runtime)


def train_model(x, y, seed=config.SEED, cache_name=None, scaler=None):
    """Fit the forest on the original graph. y is the PageRank of every node.

    The model keeps the column order it was trained with, so predict_scores
    can check that it gets the same columns later.
    """
    model = RandomForestRegressor(random_state=seed, **FOREST_SETTINGS)
    model.fit(x, y)
    model._reproducibility_fingerprint = cache_metadata.stable_fingerprint(
        _model_spec(cache_name, x, y, seed, scaler))
    return model


def predict_scores(model, x):
    """Score the nodes of any graph with the already trained model (no retraining).

    Prediction runs on one thread. With several threads the 200 tree outputs
    are added up in a different order each time, and the result changes in
    the 16th digit - enough to break "same seed, identical output" (found by
    tests/test_pipeline.py). One thread costs ~0.1 s more per graph.
    """
    columns = list(model.feature_names_in_)
    n_jobs = model.n_jobs
    model.n_jobs = 1
    try:
        prediction = model.predict(x[columns])
    finally:
        model.n_jobs = n_jobs
    return pd.Series(prediction, index=x.index)


def holdout_check(x, y, seed=config.SEED):
    """Sanity check only: train on 80 % of the papers, test on the other 20 %.

    The stability experiments use the model trained on all nodes (as in
    Phase A); this just tells us whether the forest learned anything real.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    model = train_model(x_train, y_train, seed)
    pred = predict_scores(model, x_test)
    return {
        "r2": r2_score(y_test, pred),
        "spearman": pred.corr(y_test, method="spearman"),
    }


def save_model_cache(name, forest, x, y, models_dir, seed=config.SEED, scaler=None):
    """Save a fitted forest plus the metadata required to validate later reuse."""
    import joblib

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"rf_{name}.joblib"
    spec = _model_spec(name, x, y, seed, scaler)
    forest._reproducibility_fingerprint = cache_metadata.stable_fingerprint(spec)
    joblib.dump(forest, path)
    cache_metadata.write_metadata(path, spec)
    return path


def load_or_train(name, x, y, models_dir, seed=config.SEED, scaler=None, rebuild_invalid=False):
    """Load a validated forest, or build one when absent.

    An existing invalid cache is preserved and rejected unless the caller gives
    explicit confirmation with ``rebuild_invalid=True``.
    """
    import joblib

    path = Path(models_dir) / f"rf_{name}.joblib"
    spec = _model_spec(name, x, y, seed, scaler)
    if path.exists():
        try:
            payload = cache_metadata.validate_metadata(path, spec)
            forest = joblib.load(path)
            forest._reproducibility_fingerprint = payload["spec_fingerprint"]
            return forest
        except (cache_metadata.CacheValidationError, OSError, ValueError) as error:
            if not rebuild_invalid:
                raise cache_metadata.CacheValidationError(
                    f"model cache {path} is invalid; preserve it and rerun with rebuild_invalid=True to replace it"
                ) from error
    forest = train_model(x, y, seed, cache_name=name, scaler=scaler)
    save_model_cache(name, forest, x, y, models_dir, seed, scaler)
    return forest
