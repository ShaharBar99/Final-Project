import networkx as nx
import numpy as np
import pytest

from src import experiment, features, model, perturb, scaling


@pytest.fixture(scope="module")
def setup():
    g = nx.gnp_random_graph(120, 0.04, seed=7, directed=True)
    g = nx.DiGraph((str(a), str(b)) for a, b in g.edges if a > b)  # make it a DAG
    raw = features.compute_features(g)
    scaler = scaling.fit_scaler(raw)
    x = scaling.apply_scaler(scaler, raw)
    forests = {name: model.train_model(x[cols], x[model.TARGET])
               for name, cols in experiment.VARIANTS.items()}
    return g, scaler, forests


def test_scores_have_all_methods_and_papers(setup):
    g, scaler, forests = setup
    scores, out_of_range = experiment.score_graph(g, scaler, forests)
    assert list(scores.columns) == experiment.METHODS
    assert set(scores.index) == set(g.nodes)
    # on the graph the scaler was fitted on, nothing can be out of range
    assert all(v == 0 for v in out_of_range.values())


def test_unit_conversion_undoes_the_scaler(setup):
    g, scaler, _ = setup
    raw = features.compute_features(g)
    scaled = scaling.apply_scaler(scaler, raw)
    back = experiment.to_pagerank_units(scaled["pagerank"], scaler)
    assert np.allclose(back, raw["pagerank"])


def test_forest_scores_are_positive_and_follow_pagerank(setup):
    g, scaler, forests = setup
    scores, _ = experiment.score_graph(g, scaler, forests)
    assert (scores[["rf_with_pagerank", "rf_without_pagerank"]] > 0).all().all()
    assert scores["rf_with_pagerank"].corr(scores["pagerank"], method="spearman") > 0.9


def test_zero_damage_gives_identical_runs(setup):
    g, scaler, forests = setup
    result, info = experiment.run_setting(g, "remove", 0.0, 3, scaler, forests, order=None)
    base, _ = experiment.score_graph(g, scaler, forests)
    for run in range(3):
        assert np.allclose(result[run], base.loc[list(g.nodes)].to_numpy())


def test_damage_changes_scores(setup):
    g, scaler, forests = setup
    order = perturb.time_order(g)
    result, _ = experiment.run_setting(g, "add_dag", 0.2, 2, scaler, forests, order)
    assert not np.allclose(result[0], result[1])
