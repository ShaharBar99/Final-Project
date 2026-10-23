import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src import features, scaling


def small_graph():
    # 1 cites 2 and 3, 2 cites 3, 4 cites 2, node 5 is only cited by 4
    g = nx.DiGraph([("1", "2"), ("1", "3"), ("2", "3"), ("4", "2"), ("4", "5")])
    return g


def test_table_shape_and_columns():
    t = features.compute_features(small_graph())
    assert list(t.columns) == features.FEATURE_NAMES
    assert len(t) == 5
    assert list(t.index) == list(small_graph().nodes)


def test_degree_matches_definition():
    t = features.compute_features(small_graph())
    # node 2: in-degree 2 (from 1 and 4) + out-degree 1, divided by N-1 = 4
    assert t.loc["2", "degree"] == pytest.approx(3 / 4)


def test_betweenness_agrees_with_networkx():
    g = small_graph()
    ours = features.compute_features(g)["betweenness"]
    theirs = pd.Series(nx.betweenness_centrality(g))
    assert np.allclose(ours.sort_index(), theirs.sort_index())


def test_betweenness_agrees_with_networkx_on_random_graph():
    g = nx.gnp_random_graph(60, 0.06, seed=3, directed=True)
    g = nx.relabel_nodes(g, str)
    ours = features.compute_features(g)["betweenness"]
    theirs = pd.Series(nx.betweenness_centrality(g))
    assert np.allclose(ours.sort_index(), theirs.sort_index())


def test_node_that_reaches_nobody_has_zero_closeness():
    g = nx.DiGraph([("a", "b")])
    g.add_node("c")  # not connected to anything
    t = features.compute_features(g)
    assert t.loc["c", "closeness"] == 0


def test_kcore_ignores_self_loops():
    g = nx.DiGraph([("a", "b"), ("b", "c"), ("a", "a")])
    t = features.compute_features(g)
    assert t.loc["a", "kcore"] == 1


def test_pagerank_sums_to_one():
    t = features.compute_features(small_graph())
    assert t["pagerank"].sum() == pytest.approx(1.0)


def test_scaler_is_bounded_on_original_graph():
    t = features.compute_features(small_graph())
    s = scaling.apply_scaler(scaling.fit_scaler(t), t)
    assert s.min().min() >= 0 and s.max().max() <= 1


def test_scaler_is_frozen_and_not_clipped():
    original = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    scaler = scaling.fit_scaler(original)
    perturbed = pd.DataFrame({"x": [0.0, 4.0]})
    out = scaling.apply_scaler(scaler, perturbed)
    assert out["x"].tolist() == [0.0, 2.0]  # 4 on a 0..2 scale -> 2, not clipped to 1


def test_closeness_agrees_with_networkx_on_random_graph_with_cycles():
    g = nx.gnp_random_graph(80, 0.04, seed=5, directed=True)
    g = nx.relabel_nodes(g, str)
    g.add_node("lonely")
    ours = features.compute_features(g)["closeness"]
    theirs = pd.Series(nx.closeness_centrality(g))
    assert np.allclose(ours.sort_index(), theirs.sort_index())
