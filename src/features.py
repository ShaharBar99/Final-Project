"""Block A - structural features (the five centralities) of a citation graph."""
import igraph as ig
import networkx as nx
import pandas as pd

FEATURE_NAMES = ["degree", "betweenness", "closeness", "pagerank", "kcore"]

DAMPING = 0.85  # the usual PageRank value, same as in Appendix B of Phase A


def _betweenness(g, nodes):
    """Exact directed betweenness, divided by (n-1)(n-2) like networkx does.

    networkx needs ~3 minutes for this on our graph, igraph does it in under a
    second, so we use igraph here and keep networkx for everything else.
    The result is checked against networkx in tests/test_features.py.
    """
    index = {node: i for i, node in enumerate(nodes)}
    edges = [(index[a], index[b]) for a, b in g.edges]
    h = ig.Graph(n=len(nodes), edges=edges, directed=True)
    raw = h.betweenness(directed=True)
    n = len(nodes)
    scale = (n - 1) * (n - 2) if n > 2 else 1
    return pd.Series(raw, index=nodes) / scale


def compute_features(g):
    """Return a DataFrame (index = paper_id, columns = FEATURE_NAMES) for graph g.

    The rows follow the node order of g, so graphs that share the same node
    set (the original and all perturbed copies) give tables that line up.
    """
    nodes = list(g.nodes)

    # k-core needs an undirected graph without self-loops
    undirected = nx.Graph(g)
    undirected.remove_edges_from(nx.selfloop_edges(undirected))

    table = pd.DataFrame(index=nodes)
    table["degree"] = pd.Series(nx.degree_centrality(g))
    table["betweenness"] = _betweenness(g, nodes)
    # for directed graphs networkx measures *incoming* distance, and corrects
    # for nodes that cannot be reached from everywhere
    table["closeness"] = pd.Series(nx.closeness_centrality(g))
    table["pagerank"] = pd.Series(nx.pagerank(g, alpha=DAMPING))
    table["kcore"] = pd.Series(nx.core_number(undirected))
    return table[FEATURE_NAMES]
