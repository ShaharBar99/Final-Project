"""Block A - structural features (the five centralities) of a citation graph."""
import igraph as ig
import networkx as nx
import pandas as pd

FEATURE_NAMES = ["degree", "betweenness", "closeness", "pagerank", "kcore"]

DAMPING = 0.85  # the usual PageRank value, same as in Appendix B of Phase A


def _to_igraph(g, nodes):
    """Same graph as an igraph object; vertex i is nodes[i]."""
    index = {node: i for i, node in enumerate(nodes)}
    edges = [(index[a], index[b]) for a, b in g.edges]
    return ig.Graph(n=len(nodes), edges=edges, directed=True)


def _betweenness(h, nodes):
    """Exact directed betweenness, divided by (n-1)(n-2) like networkx does.

    networkx needs ~3 minutes for this on our graph, igraph does it in under a
    second, so the two path-based measures are computed with igraph.
    Both are checked against networkx in tests/test_features.py.
    """
    n = len(nodes)
    scale = (n - 1) * (n - 2) if n > 2 else 1
    return pd.Series(h.betweenness(directed=True), index=nodes) / scale


def _closeness(h, nodes):
    """Closeness on incoming distances, same formula as networkx (Wasserman-Faust).

    For a paper u, let r = number of papers that can reach u. Then
        closeness(u) = (r / sum of distances) * (r / (n - 1))
    The second factor punishes papers that only a few others can reach.
    A paper that nobody reaches gets 0.

    networkx is fine on the original graph (0.5 s), but on graphs with random
    edges, cycles appear and it takes ~20 s, so igraph is used here as well.
    """
    n = len(nodes)
    # igraph gives r / sum of distances (NaN when r = 0)
    base = pd.Series(h.closeness(mode="in", normalized=True), index=nodes).fillna(0.0)
    reach = pd.Series(h.neighborhood_size(order=n, mode="in"), index=nodes) - 1
    return base * reach / (n - 1) if n > 1 else base * 0


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
    h = _to_igraph(g, nodes)
    table["betweenness"] = _betweenness(h, nodes)
    table["closeness"] = _closeness(h, nodes)
    table["pagerank"] = pd.Series(nx.pagerank(g, alpha=DAMPING))
    table["kcore"] = pd.Series(nx.core_number(undirected))
    return table[FEATURE_NAMES]
