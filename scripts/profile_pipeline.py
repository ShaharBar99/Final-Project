"""Performance and memory profiling (Phase A, Section 7.2 - performance and load testing).

Three measurements, written to results/profiling/:
    stages.csv       time and memory of every pipeline stage on the full graph
    k_runs.csv       time and memory for K = 1, 10, 30 perturbed graphs (does memory grow?)
    scalability.csv  time per perturbed graph for growing parts of the graph

Run from the project root:
    python scripts/profile_pipeline.py
Takes about 3 minutes.
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data, demo, experiment, features, model, perturb, scaling  # noqa: E402

OUT = config.RESULTS_DIR / "profiling"
PROCESS = psutil.Process()


def rss_mb():
    return PROCESS.memory_info().rss / 1e6


def measure(name, func, rows):
    """Run func once, record its time and the memory of the process after it."""
    gc.collect()
    before = rss_mb()
    start = time.perf_counter()
    result = func()
    rows.append({"stage": name, "seconds": round(time.perf_counter() - start, 3),
                 "memory_after_mb": round(rss_mb(), 1),
                 "memory_change_mb": round(rss_mb() - before, 1)})
    print(rows[-1], flush=True)
    return result


def profile_stages():
    rows = []
    g, nodes = measure("load cleaned graph", data.load_clean_graph, rows)
    raw = measure("compute 5 features", lambda: features.compute_features(g), rows)
    scaler = measure("fit scaler", lambda: scaling.fit_scaler(raw), rows)
    x = measure("apply scaler", lambda: scaling.apply_scaler(scaler, raw), rows)
    forests = measure("train both forests", lambda: {
        name: model.train_model(x[cols], x[model.TARGET]) for name, cols in experiment.VARIANTS.items()}, rows)
    measure("predict both forests", lambda: [model.predict_scores(f, x) for f in forests.values()], rows)
    order = measure("topological order", lambda: perturb.time_order(g), rows)
    for kind in perturb.KINDS:
        damaged = measure(f"perturb {kind} 10 %", lambda: perturb.perturb(
            g, kind, 0.10, perturb.run_rng(config.SEED, kind, 0.10, 0), order), rows)
        measure(f"score graph after {kind} 10 %", lambda: experiment.score_graph(damaged, scaler, forests), rows)
    pd.DataFrame(rows).to_csv(OUT / "stages.csv", index=False)
    return g, scaler, forests, order


def profile_k(g, scaler, forests, order):
    """Memory should grow only by the size of the score array (K x papers x 7 x 8 bytes)."""
    rows = []
    for k in [1, 10, 30]:
        gc.collect()
        before = rss_mb()
        peak = before
        start = time.perf_counter()
        scores = np.empty((k, g.number_of_nodes(), len(experiment.METHODS)))
        for run in range(k):
            damaged = perturb.perturb(g, "remove", 0.10, perturb.run_rng(config.SEED, "remove", 0.10, run), order)
            s, _ = experiment.score_graph(damaged, scaler, forests)
            scores[run] = s.to_numpy()
            peak = max(peak, rss_mb())
            del damaged, s
        rows.append({"K": k, "seconds": round(time.perf_counter() - start, 1),
                     "score_array_mb": round(scores.nbytes / 1e6, 1),
                     "memory_growth_mb": round(rss_mb() - before, 1),
                     "peak_above_start_mb": round(peak - before, 1)})
        print(rows[-1], flush=True)
        del scores
    pd.DataFrame(rows).to_csv(OUT / "k_runs.csv", index=False)


def profile_scalability(g, scaler, forests):
    """Time to perturb and score one graph, for random parts of the graph (induced subgraphs)."""
    rows = []
    rng = np.random.default_rng(config.SEED)
    all_nodes = list(g.nodes)
    for share in [0.25, 0.5, 0.75, 1.0]:
        picked = rng.choice(len(all_nodes), size=int(share * len(all_nodes)), replace=False)
        part = g.subgraph(all_nodes[i] for i in picked).copy()
        order = perturb.time_order(part)
        for kind in perturb.KINDS:
            damaged = perturb.perturb(part, kind, 0.10, perturb.run_rng(config.SEED, kind, 0.10, 0), order)
            start = time.perf_counter()
            experiment.score_graph(damaged, scaler, forests)
            rows.append({"share_of_papers": share, "papers": part.number_of_nodes(),
                         "edges": part.number_of_edges(), "kind": kind,
                         "seconds_per_graph": round(time.perf_counter() - start, 2)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "scalability.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"start: {rss_mb():.0f} MB")
    g, scaler, forests, order = profile_stages()
    profile_k(g, scaler, forests, order)
    profile_scalability(g, scaler, forests)
    print("done")
