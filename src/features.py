"""Block A - structural features (the five centralities) of a citation graph."""

FEATURE_NAMES = ["degree", "betweenness", "closeness", "pagerank", "kcore"]


def compute_features(g, betweenness_k=500, seed=None):
    """Return a DataFrame (index = paper_id, columns = FEATURE_NAMES) for graph g."""
    raise NotImplementedError("step 3")
