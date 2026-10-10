import pandas as pd

from src import data


def make_edges(pairs):
    return pd.DataFrame(pairs, columns=["source", "target"])


def test_duplicates_are_removed():
    log = []
    edges = data.remove_duplicate_edges(make_edges([("1", "2"), ("1", "2"), ("2", "3")]), log)
    assert len(edges) == 2
    assert log[0]["removed"] == 1


def test_self_loops_are_removed():
    log = []
    edges = data.remove_self_loops(make_edges([("1", "1"), ("1", "2")]), log)
    assert edges.values.tolist() == [["1", "2"]]


def test_mutual_citation_keeps_newer_to_older():
    log = []
    edges = data.resolve_mutual_citations(make_edges([("5", "9"), ("9", "5"), ("9", "1")]), log)
    kept = set(map(tuple, edges.values.tolist()))
    assert kept == {("9", "5"), ("9", "1")}


def test_isolated_nodes_are_removed():
    log = []
    nodes = pd.DataFrame({"paper_id": ["1", "2", "3"], "label": [1, 2, 3]})
    nodes = data.remove_isolated_nodes(nodes, make_edges([("1", "2")]), log)
    assert nodes["paper_id"].tolist() == ["1", "2"]


def test_full_pipeline_gives_a_clean_graph():
    nodes, edges, log = data.clean_dataset()
    assert (edges["source"] != edges["target"]).all()
    assert not edges.duplicated().any()
    final = log[-1]
    assert final["directed_cycles_groups"] == 0
    assert final["nodes"] == len(nodes)
