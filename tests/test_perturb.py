import networkx as nx
import numpy as np
import pytest

from src import perturb


def chain_graph(n=200):
    # a DAG: every paper cites the two papers before it
    g = nx.DiGraph()
    for i in range(2, n):
        g.add_edge(str(i), str(i - 1))
        g.add_edge(str(i), str(i - 2))
    return g


@pytest.mark.parametrize("kind", perturb.KINDS)
def test_nodes_never_change(kind):
    g = chain_graph()
    damaged = perturb.perturb(g, kind, 0.2, np.random.default_rng(0))
    assert set(damaged.nodes) == set(g.nodes)


def test_original_graph_is_not_modified():
    g = chain_graph()
    before = set(g.edges)
    perturb.perturb(g, "remove", 0.3, np.random.default_rng(0))
    perturb.perturb(g, "add_random", 0.3, np.random.default_rng(0))
    assert set(g.edges) == before


def test_remove_deletes_the_right_number_of_existing_edges():
    g = chain_graph()
    damaged = perturb.remove_edges(g, 0.1, np.random.default_rng(0))
    assert damaged.number_of_edges() == g.number_of_edges() - round(0.1 * g.number_of_edges())
    assert set(damaged.edges) <= set(g.edges)


@pytest.mark.parametrize("kind", ["add_random", "add_dag"])
def test_add_keeps_old_edges_and_adds_new_ones(kind):
    g = chain_graph()
    damaged = perturb.perturb(g, kind, 0.2, np.random.default_rng(0))
    assert set(g.edges) <= set(damaged.edges)
    assert damaged.number_of_edges() == g.number_of_edges() + round(0.2 * g.number_of_edges())
    assert nx.number_of_selfloops(damaged) == 0


def test_add_dag_keeps_the_graph_acyclic():
    g = chain_graph()
    damaged = perturb.perturb(g, "add_dag", 0.5, np.random.default_rng(0))
    assert nx.is_directed_acyclic_graph(damaged)


def test_add_random_can_create_cycles():
    # with enough random edges on a DAG, some of them point "forward in time"
    g = chain_graph()
    damaged = perturb.perturb(g, "add_random", 0.5, np.random.default_rng(0))
    assert not nx.is_directed_acyclic_graph(damaged)


def test_same_seed_same_graph_different_run_different_graph():
    g = chain_graph()
    a = perturb.perturb(g, "remove", 0.2, perturb.run_rng(42, "remove", 0.2, 0))
    b = perturb.perturb(g, "remove", 0.2, perturb.run_rng(42, "remove", 0.2, 0))
    c = perturb.perturb(g, "remove", 0.2, perturb.run_rng(42, "remove", 0.2, 1))
    assert set(a.edges) == set(b.edges)
    assert set(a.edges) != set(c.edges)
