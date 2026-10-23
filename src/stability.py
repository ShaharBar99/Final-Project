"""Block F - how much do scores and rankings move under perturbation.

Two views, as in Phase A (Sections 4.4, 4.7, 4.8):
    node level  - coefficient of variation (CV) of each paper's score over the K runs
    ranking     - Spearman and Kendall between the base ranking and each run's ranking
plus top-k retention (does the top of the ranking survive?) for the core question.
"""
import numpy as np
import pandas as pd
from scipy import stats

from . import ranking


def node_instability(runs):
    """CV = std / mean of every paper over the runs. `runs` is (K runs x papers).

    Population std (ddof=0). If a paper's mean score is 0 its CV is set to 0,
    as defined in Phase A, Section 4.4 - such papers are counted separately by
    zero_mean_share() because a forced 0 is not the same as "stable".
    """
    runs = np.asarray(runs, dtype=float)
    mean = runs.mean(axis=0)
    std = runs.std(axis=0)
    cv = np.zeros_like(mean)
    nonzero = mean > 0
    cv[nonzero] = std[nonzero] / mean[nonzero]
    return cv


def zero_mean_share(runs):
    """Share of papers whose score is 0 in every run (their CV is 0 by definition)."""
    return float((np.asarray(runs).mean(axis=0) == 0).mean())


def rank_correlations(base_ranks, run_ranks):
    """Spearman and Kendall (tau-b, handles ties) between two rankings of the same papers."""
    spearman = stats.spearmanr(base_ranks, run_ranks).statistic
    kendall = stats.kendalltau(base_ranks, run_ranks).statistic
    return spearman, kendall


def top_k_retention(base_ranks, run_ranks, k):
    """Share of the papers in the base top-k that are still in the top-k of the run.

    "Top-k" = rank <= k. With ties (average ranks) a tied group can make the
    base top-k a bit larger or smaller than k; the share is taken over the
    papers that really are in the base top-k.
    """
    base_top = base_ranks <= k
    if base_top.sum() == 0:
        return np.nan
    return float((run_ranks[base_top] <= k).mean())


def ranking_summary(correlations):
    """Mean, std and ranking instability (std / mean) of a vector of correlations (Phase A 4.8)."""
    c = np.asarray(correlations, dtype=float)
    mean, std = c.mean(), c.std()
    return {"mean": mean, "std": std, "instability": std / mean if mean != 0 else np.nan}


def evaluate_setting(scores, base_scores, methods, top_ks=(100, 1000), top_share=0.10):
    """All stability numbers for one (kind, level).

    scores      : array (K runs x papers x methods), natural units
    base_scores : DataFrame (papers x methods) on the original graph, same paper order
    Returns (per_run, cv):
        per_run  DataFrame, one row per (method, run): spearman, kendall,
                 spearman_top10pct, top-k retention
        cv       DataFrame (papers x methods) with each paper's CV

    spearman_top10pct is Spearman over the method's own base top 10 % only.
    It exists because ties inflate the full correlation: if 90 % of papers sit
    at 0 in every run (betweenness), they stay tied and the correlation looks
    high even when the ranking among the papers that matter moves a lot.
    """
    rows = []
    cv = {}
    for m, method in enumerate(methods):
        base_ranks = ranking.to_ranking(base_scores[method]).to_numpy()
        top = base_ranks <= top_share * len(base_ranks)
        for run in range(scores.shape[0]):
            run_ranks = ranking.to_ranking(pd.Series(scores[run, :, m])).to_numpy()
            spearman, kendall = rank_correlations(base_ranks, run_ranks)
            top_spearman = stats.spearmanr(base_ranks[top], run_ranks[top]).statistic
            row = {"method": method, "run": run, "spearman": spearman, "kendall": kendall,
                   "spearman_top10pct": top_spearman}
            for k in top_ks:
                row[f"top{k}_kept"] = top_k_retention(base_ranks, run_ranks, k)
            rows.append(row)
        cv[method] = node_instability(scores[:, :, m])
    return pd.DataFrame(rows), pd.DataFrame(cv, index=base_scores.index)


def cv_summary(cv, base_scores, zero_share, top_share=0.10):
    """Per method: mean / median CV over all papers and over the method's own top 10 %."""
    rows = {}
    for method in cv.columns:
        ranks = ranking.to_ranking(base_scores[method])
        top = ranks <= top_share * len(ranks)
        rows[method] = {
            "mean_cv": cv[method].mean(),
            "median_cv": cv[method].median(),
            "mean_cv_top10pct": cv[method][top].mean(),
            "zero_mean_share": zero_share[method],
        }
    return pd.DataFrame(rows).T


def instability_profile_correlation(cv, reference):
    """Spearman between the CV vector of `reference` and that of every other method (Phase A 4.7).

    Close to 1: the reference is unstable on the same papers as that measure.
    Close to 0: independent. Negative: it is stable where the measure is not.
    """
    return {m: stats.spearmanr(cv[reference], cv[m]).statistic
            for m in cv.columns if m != reference}
