import pandas as pd

from src import core


def test_roles_split_by_importance_and_stability():
    # 10 papers: rank 1 is the most important; cv decides stability
    ranks = pd.Series(range(1, 11), index=list("abcdefghij"), dtype=float)
    cv = pd.Series([0.0, 0.9, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9], index=ranks.index)
    roles, threshold = core.node_roles(ranks, cv, top_share=0.2, cv_percentile=40)
    assert threshold == 0.0
    assert roles["a"] == "core"                   # rank 1, stable
    assert roles["b"] == "important but unstable"  # rank 2, unstable
    assert roles["c"] == "stable periphery"
    assert roles["d"] == "unstable periphery"
    assert core.find_core(ranks, cv, 0.2, 40) == {"a"}


def test_jaccard():
    assert core.jaccard({1, 2}, {2, 3}) == 1 / 3
    assert core.jaccard(set(), set()) == 1.0


def test_adjusted_core_splits_the_important_papers_by_median_cv():
    ranks = pd.Series(range(1, 11), index=list("abcdefghij"), dtype=float)
    cv = pd.Series([0.1, 0.5, 0.2, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=ranks.index)
    roles, threshold = core.node_roles_within_important(ranks, cv, top_share=0.4)
    assert threshold == 0.35  # median of 0.1, 0.5, 0.2, 0.9
    assert roles[["a", "c"]].eq("core").all()
    assert roles[["b", "d"]].eq("important but unstable").all()
    assert roles[list("efghij")].eq("periphery").all()


def test_stay_in_top_share():
    import numpy as np
    ranks = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
    runs = np.array([[4, 3, 2, 1], [4, 3, 2, 1]], dtype=float)  # a stays first, d falls
    share = core.stay_in_top_share(ranks, runs, top_share=0.25)
    assert share.tolist() == [1.0, 0.0, 0.0, 0.0]
