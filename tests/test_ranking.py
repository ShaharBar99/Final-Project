import pandas as pd

from src import ranking


def test_highest_score_gets_rank_one():
    r = ranking.to_ranking(pd.Series([0.1, 0.9, 0.5], index=["a", "b", "c"]))
    assert r.to_dict() == {"a": 3.0, "b": 1.0, "c": 2.0}


def test_ties_share_the_average_rank():
    r = ranking.to_ranking(pd.Series([1.0, 1.0, 0.0, 0.0, 0.0]))
    assert r.tolist() == [1.5, 1.5, 4.0, 4.0, 4.0]


def test_ranking_keeps_the_index():
    s = pd.Series([3, 1], index=["x", "y"])
    assert list(ranking.to_ranking(s).index) == ["x", "y"]


def test_tie_share():
    assert ranking.tie_share(pd.Series([1, 1, 2, 3])) == 0.5
