"""Integration and end-to-end tests (Phase A, Section 7.2).

The first group runs the whole pipeline on a small synthetic citation graph, so
it is fast and needs no data files. The tests marked `slow` use the real
cleaned data in data/processed/ (skip them with: pytest -m "not slow").
"""
import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src import config, core, experiment, features, model, perturb, ranking, scaling, stability


def synthetic_citations(n=300, seed=0):
    """A small DAG that looks a bit like a citation graph: new papers cite older, popular ones."""
    rng = np.random.default_rng(seed)
    g = nx.DiGraph()
    g.add_node("0")
    for new in range(1, n):
        in_deg = np.array([g.in_degree(str(old)) + 1 for old in range(new)], dtype=float)
        k = min(new, rng.integers(1, 4))
        cited = rng.choice(new, size=k, replace=False, p=in_deg / in_deg.sum())
        g.add_edges_from((str(new), str(old)) for old in cited)
    return g


def run_pipeline(g, k=5, seed=1, kinds=("remove", "add_dag")):
    """Everything from features to core, the way the real experiment does it."""
    raw = features.compute_features(g)
    scaler = scaling.fit_scaler(raw)
    x = scaling.apply_scaler(scaler, raw)
    forests = {name: model.train_model(x[cols], x[model.TARGET], seed=seed)
               for name, cols in experiment.VARIANTS.items()}
    base, _ = experiment.score_graph(g, scaler, forests)
    base = base.loc[list(g.nodes)]
    order = perturb.time_order(g)
    out = {}
    for kind in kinds:
        scores, _ = experiment.run_setting(g, kind, 0.1, k, scaler, forests, order, seed=seed)
        per_run, cv = stability.evaluate_setting(scores, base, experiment.METHODS, top_ks=(10,))
        out[kind] = (scores, per_run, cv)
    return base, out


@pytest.fixture(scope="module")
def small():
    g = synthetic_citations()
    return g, run_pipeline(g)


# ---------------------------------------------------------------- integration

def test_synthetic_graph_is_a_dag(small):
    g, _ = small
    assert nx.is_directed_acyclic_graph(g)


def test_all_stages_line_up_on_the_same_papers(small):
    g, (base, out) = small
    assert list(base.index) == list(g.nodes)
    for scores, per_run, cv in out.values():
        assert scores.shape == (5, g.number_of_nodes(), len(experiment.METHODS))
        assert list(cv.index) == list(g.nodes)
        assert len(per_run) == 5 * len(experiment.METHODS)
        assert np.isfinite(scores).all()


def test_correlations_are_valid_numbers(small):
    _, (_, out) = small
    for _, per_run, _ in out.values():
        assert per_run[["spearman", "kendall"]].abs().le(1).all().all()


def test_cv_is_never_negative(small):
    _, (_, out) = small
    for _, _, cv in out.values():
        assert (cv.to_numpy() >= 0).all()


# ---------------------------------------------------------------- end to end

def test_same_seed_gives_identical_results():
    g = synthetic_citations(150)
    base_a, out_a = run_pipeline(g, k=3, seed=7)
    base_b, out_b = run_pipeline(g, k=3, seed=7)
    assert np.array_equal(base_a.to_numpy(), base_b.to_numpy())
    for kind in out_a:
        assert np.array_equal(out_a[kind][0], out_b[kind][0])


def test_different_seed_gives_different_damage():
    g = synthetic_citations(150)
    _, out_a = run_pipeline(g, k=2, seed=7, kinds=("remove",))
    _, out_b = run_pipeline(g, k=2, seed=8, kinds=("remove",))
    assert not np.array_equal(out_a["remove"][0], out_b["remove"][0])


def test_core_threshold_classifies_correctly(small):
    """Every core paper is important and at or below the CV threshold, and nobody else is core."""
    _, (base, out) = small
    cv = out["add_dag"][2]["rf_without_pagerank"]
    ranks = ranking.to_ranking(base["rf_without_pagerank"])
    roles, threshold = core.node_roles_within_important(ranks, cv)
    important = ranks <= 0.1 * len(ranks)
    expected_core = important & (cv <= threshold)
    assert (roles == "core").equals(expected_core)
    assert ((roles == "important but unstable") == (important & (cv > threshold))).all()


# ---------------------------------------------------------------- real data

@pytest.mark.slow
def test_real_data_pipeline_reproduces_saved_base_ranking():
    from src import demo
    g, _, scaler, forests, base = demo.load_pipeline()
    saved = pd.read_csv(config.PROCESSED_DIR / "base_ranking.csv", index_col="paper_id",
                        dtype={"paper_id": str})
    for m in experiment.METHODS:
        assert (ranking.to_ranking(base[m]) == saved.loc[base.index, m]).all(), m


@pytest.mark.slow
def test_real_data_demo_is_reproducible_and_matches_the_experiment():
    from src import demo
    g, _, scaler, forests, _ = demo.load_pipeline()
    _, a, _ = demo.damage_and_score(g, scaler, forests, "remove", 0.1)
    _, b, _ = demo.damage_and_score(g, scaler, forests, "remove", 0.1)
    assert np.array_equal(a.to_numpy(), b.to_numpy())
    saved = experiment.RUNS_DIR / "remove_10.npz"
    if saved.exists():  # the raw runs are not on GitHub; only checked when they were built locally
        runs, nodes, _ = experiment.load_setting("remove", 0.1)
        assert np.allclose(runs[0], a.loc[nodes].to_numpy())


@pytest.mark.slow
def test_app_runs_without_errors():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(config.ROOT / "app" / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception
    app.sidebar.radio[0].set_value("remove").run()
    assert not app.exception
