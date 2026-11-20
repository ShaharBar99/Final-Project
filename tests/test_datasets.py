import networkx as nx
import pandas as pd

from src import config, data, features


def make_edges(pairs):
    return pd.DataFrame(pairs, columns=["source", "target"])


def test_cycle_breaking_leaves_a_dag_and_removes_few_edges():
    # a mutual pair (1,2), a loop 3 -> 4 -> 5 -> 3, and two edges that are fine
    edges = make_edges([("1", "2"), ("2", "1"), ("3", "4"), ("4", "5"), ("5", "3"), ("6", "1"), ("5", "7")])
    log = []
    cleaned = data.break_citation_cycles(edges, log)
    g = nx.from_pandas_edgelist(cleaned, "source", "target", create_using=nx.DiGraph)
    assert nx.is_directed_acyclic_graph(g)
    assert log[0]["removed"] == 2          # one edge of the pair, one edge of the loop - the minimum
    assert log[0]["mutual_pairs_before"] == 1
    assert log[0]["loop_groups_before"] == 2
    assert ("6", "1") in set(map(tuple, cleaned.values.tolist()))


def test_cycle_breaking_does_nothing_on_a_dag():
    edges = make_edges([("1", "2"), ("2", "3"), ("1", "3")])
    log = []
    assert len(data.break_citation_cycles(edges, log)) == 3
    assert log[0]["removed"] == 0


def test_linqs_links_go_from_the_citing_to_the_cited_paper(tmp_path, monkeypatch):
    # the file lists "<cited> <citing>"; the graph needs citing -> cited
    folder = tmp_path / "toy"
    folder.mkdir()
    (folder / "toy.nodes.tab").write_text("paper_id\tlabel\nA\tx\nB\ty\n", encoding="utf-8")
    (folder / "toy.cites").write_text("A\tB\n", encoding="utf-8")        # B cites A
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    nodes, edges = data.load_linqs("toy")
    assert edges.values.tolist() == [["B", "A"]]
    assert nodes["paper_id"].tolist() == ["A", "B"]


def test_cora_and_citeseer_are_clean_dags():
    for name, papers in (("cora", 2708), ("citeseer", 3264)):
        nodes, edges, log = data.clean_dataset(name)
        g = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.DiGraph)
        assert nx.is_directed_acyclic_graph(g)
        assert len(nodes) == papers
        assert (edges["source"] != edges["target"]).all()
        assert not edges.duplicated().any()
        assert set(edges["source"]) | set(edges["target"]) == set(nodes["paper_id"])


def test_feature_set_without_closeness_drops_the_column(monkeypatch):
    monkeypatch.setattr(features, "FEATURE_NAMES", ["degree", "betweenness", "pagerank", "kcore"])
    g = nx.DiGraph([("1", "2"), ("1", "3"), ("2", "3"), ("4", "2"), ("4", "5")])
    table = features.compute_features(g)
    assert list(table.columns) == ["degree", "betweenness", "pagerank", "kcore"]
