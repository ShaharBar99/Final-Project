"""Step 7 - run every method on every perturbed graph and keep the scores.

For one perturbed graph:
    features (block A) -> frozen scaler (B) -> both forests, no retraining (C)
and the single centralities are kept as they are.

Scores are stored in natural units (decision D12):
    single centralities - their raw value
    forests             - their prediction turned back into PageRank units
This matters only for the coefficient of variation, which changes when values
are shifted (Min-Max subtracts the minimum). Rankings are not affected.
"""
import json
import time

import numpy as np
import pandas as pd

from . import cache_metadata, config, features, model, perturb, scaling

VARIANTS = {
    "with_pagerank": model.FEATURES_WITH_PAGERANK,
    "without_pagerank": model.FEATURES_WITHOUT_PAGERANK,
}
METHODS = [f"rf_{name}" for name in VARIANTS] + features.FEATURE_NAMES
LEVELS = [0.05, 0.10, 0.20]
RUNS_DIR = config.RESULTS_DIR / "runs"


def to_pagerank_units(scaled_prediction, scaler):
    """Undo the Min-Max scaling of the PageRank column for a forest prediction."""
    i = list(scaler.feature_names_in_).index(model.TARGET)
    return scaled_prediction * scaler.data_range_[i] + scaler.data_min_[i]


def score_graph(g, scaler, forests):
    """Scores of all methods on graph g, plus how many scaled values left [0, 1].

    Returns (scores, out_of_range):
        scores        DataFrame, index = papers, columns = METHODS
        out_of_range  dict feature -> share of papers whose scaled value is
                      below 0 or above 1 (the forest never saw such values)
    """
    raw = features.compute_features(g)
    scaled = scaling.apply_scaler(scaler, raw)

    scores = pd.DataFrame(index=raw.index)
    for name, forest in forests.items():
        prediction = model.predict_scores(forest, scaled)
        scores[f"rf_{name}"] = to_pagerank_units(prediction, scaler)
    for column in features.FEATURE_NAMES:
        scores[column] = raw[column]

    # tiny tolerance: on the original graph the maximum scales to 1.0000000000000002
    outside = (scaled < -1e-9) | (scaled > 1 + 1e-9)
    return scores[METHODS], outside.mean().to_dict()


def run_setting(g, kind, level, k, scaler, forests, order, seed=config.SEED):
    """K perturbed graphs of one (kind, level). Returns (array runs x papers x methods, info)."""
    nodes = list(g.nodes)
    result = np.empty((k, len(nodes), len(METHODS)), dtype=np.float64)
    info = []
    for run in range(k):
        start = time.time()
        damaged = perturb.perturb(g, kind, level, perturb.run_rng(seed, kind, level, run), order)
        scores, out_of_range = score_graph(damaged, scaler, forests)
        result[run] = scores.loc[nodes].to_numpy()
        info.append({"run": run, "edges": damaged.number_of_edges(),
                     "seconds": round(time.time() - start, 2), "out_of_range": out_of_range})
    return result, info


def setting_name(kind, level):
    return f"{kind}_{int(round(level * 100)):02d}"


def _setting_spec(g, scaler, forests, kind, level, k, seed):
    return {
        "artifact_type": "perturbation_scores",
        "graph_fingerprint": cache_metadata.graph_fingerprint(g),
        "feature_names": list(features.FEATURE_NAMES),
        "feature_definitions": features.FEATURE_DEFINITIONS,
        "model_fingerprints": {name: model.model_fingerprint(forest)
                               for name, forest in sorted(forests.items())},
        "scaler_fingerprint": cache_metadata.scaler_fingerprint(scaler),
        "scaler": cache_metadata.scaler_spec(scaler),
        "perturbation": {"kind": kind, "fraction": level, "repetitions": k, "seed": seed},
        "methods": list(METHODS),
        "dependencies": cache_metadata.dependency_versions(),
    }


def _base_spec(g, scaler, forests):
    spec = _setting_spec(g, scaler, forests, "base", 0.0, 0, config.SEED)
    spec["artifact_type"] = "base_scores"
    spec.pop("perturbation")
    return spec


def run_all(g, scaler, forests, k, kinds=perturb.KINDS, levels=LEVELS, redo=False,
            seed=config.SEED, runs_dir=None):
    """Run every setting, reusing only caches with matching validated metadata."""
    runs_dir = RUNS_DIR if runs_dir is None else runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    nodes = list(g.nodes)
    order = perturb.time_order(g)

    base_path = runs_dir / "base_scores.csv"
    base_spec = _base_spec(g, scaler, forests)
    if base_path.exists() and not redo:
        try:
            cache_metadata.validate_metadata(base_path, base_spec)
        except cache_metadata.CacheValidationError as error:
            raise cache_metadata.CacheValidationError(
                f"base-score cache {base_path} is invalid; it was preserved. "
                "Rerun with redo=True only after confirming regeneration."
            ) from error
    else:
        base, _ = score_graph(g, scaler, forests)
        base.loc[nodes].to_csv(base_path, index_label="paper_id")
        cache_metadata.write_metadata(base_path, base_spec)

    for kind in kinds:
        for level in levels:
            name = setting_name(kind, level)
            path = runs_dir / f"{name}.npz"
            spec = _setting_spec(g, scaler, forests, kind, level, k, seed)
            if path.exists() and not redo:
                try:
                    cache_metadata.validate_metadata(path, spec)
                    print(f"{name}: validated cache, skipped")
                    continue
                except cache_metadata.CacheValidationError as error:
                    raise cache_metadata.CacheValidationError(
                        f"experiment cache {path} is invalid; it was preserved. "
                        "Rerun with redo=True only after confirming regeneration."
                    ) from error
            start = time.time()
            scores, info = run_setting(g, kind, level, k, scaler, forests, order, seed=seed)
            np.savez_compressed(path, scores=scores, nodes=np.array(nodes), methods=np.array(METHODS))
            info_path = runs_dir / f"{name}_info.json"
            with open(info_path, "w") as f:
                json.dump(info, f, indent=1)
            cache_metadata.write_metadata(path, spec,
                                          {info_path.name: cache_metadata.file_sha256(info_path)})
            print(f"{name}: {k} runs in {time.time() - start:.0f} s", flush=True)


def load_setting(kind, level, runs_dir=None):
    """Load an integrity-validated setting as (scores, nodes, methods)."""
    runs_dir = RUNS_DIR if runs_dir is None else runs_dir
    path = runs_dir / f"{setting_name(kind, level)}.npz"
    cache_metadata.validate_metadata(path)
    data = np.load(path)
    return data["scores"], list(data["nodes"]), list(data["methods"])
