import numpy as np
import pandas as pd
import pytest

from src import stability


def test_cv_of_constant_scores_is_zero():
    runs = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    assert stability.node_instability(runs).tolist() == [0.0, 0.0]


def test_cv_matches_hand_calculation():
    runs = np.array([[1.0], [3.0]])  # mean 2, population std 1
    assert stability.node_instability(runs)[0] == pytest.approx(0.5)


def test_zero_mean_gives_zero_cv_and_is_counted():
    runs = np.array([[0.0, 1.0], [0.0, 2.0]])
    assert stability.node_instability(runs)[0] == 0
    assert stability.zero_mean_share(runs) == 0.5


def test_identical_rankings_correlate_perfectly():
    r = np.array([1.0, 2.0, 3.0, 4.0])
    assert stability.rank_correlations(r, r) == pytest.approx((1.0, 1.0))


def test_reversed_ranking_correlates_minus_one():
    r = np.array([1.0, 2.0, 3.0, 4.0])
    assert stability.rank_correlations(r, r[::-1]) == pytest.approx((-1.0, -1.0))


def test_top_k_retention():
    base = np.array([1, 2, 3, 4, 5], dtype=float)
    run = np.array([1, 5, 2, 3, 4], dtype=float)  # paper 2 fell out of the top 2
    assert stability.top_k_retention(base, run, 2) == 0.5


def test_evaluate_setting_with_no_change_is_perfect():
    base = pd.DataFrame({"a": [0.3, 0.2, 0.1, 0.4], "b": [1.0, 2.0, 3.0, 4.0]})
    runs = np.stack([base.to_numpy()] * 3)
    per_run, cv = stability.evaluate_setting(runs, base, ["a", "b"], top_ks=(2,))
    assert np.allclose(per_run["spearman"], 1)
    assert np.allclose(per_run["kendall"], 1)
    assert np.allclose(per_run["top2_kept"], 1)
    assert np.allclose(cv.to_numpy(), 0)
