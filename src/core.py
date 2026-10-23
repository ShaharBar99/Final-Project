"""Block G - the network core and the importance-vs-stability roles of nodes (Phase A 4.5-4.6)."""
import numpy as np
import pandas as pd
from scipy import stats

from . import ranking

ROLES = ["core", "important but unstable", "stable periphery", "unstable periphery"]


def node_roles(base_ranks, cv, top_share=0.10, cv_percentile=10):
    """Put every paper in one of four groups.

    important : in the top `top_share` of the base ranking
    stable    : CV at or below the `cv_percentile`-th percentile of all papers' CV
    core = important and stable (Phase A, Section 4.6).
    Returns (roles, threshold) - the CV threshold epsilon is returned so it can be reported.
    """
    important = base_ranks <= top_share * len(base_ranks)
    threshold = cv.quantile(cv_percentile / 100)
    stable = cv <= threshold

    roles = pd.Series("unstable periphery", index=cv.index)
    roles[~important & stable] = "stable periphery"
    roles[important & ~stable] = "important but unstable"
    roles[important & stable] = "core"
    return roles, threshold


def find_core(base_ranks, cv, top_share=0.10, cv_percentile=10):
    """Set of papers that are both important and stable."""
    roles, _ = node_roles(base_ranks, cv, top_share, cv_percentile)
    return set(roles.index[roles == "core"])


def importance_stability_correlation(base_scores, cv):
    """Spearman between a paper's base score and its CV (Phase A question 3).

    Negative: the more important a paper, the more stable it is.
    """
    return stats.spearmanr(base_scores, cv).statistic


def jaccard(a, b):
    """Overlap of two sets: |A and B| / |A or B| (1 = identical, 0 = nothing shared)."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def node_roles_within_important(base_ranks, cv, top_share=0.10):
    """Adjusted core rule (decision D16).

    The Phase A rule takes "stable" from the CV distribution of *all* papers.
    On our data the 10th percentile of that distribution is often exactly 0:
    many peripheral papers never change score at all, so the rule picks
    unimportant papers and the core can be empty.

    Here stability is judged among the important papers only:
        core                   = important and CV <= median CV of the important papers
        important but unstable = important and CV above that median
    Returns (roles, threshold); non-important papers are labelled "periphery".
    """
    important = base_ranks <= top_share * len(base_ranks)
    threshold = cv[important].median()
    roles = pd.Series("periphery", index=cv.index)
    roles[important & (cv <= threshold)] = "core"
    roles[important & (cv > threshold)] = "important but unstable"
    return roles, threshold


def stay_in_top_share(base_ranks, run_scores, top_share=0.10):
    """For every paper: share of runs in which it is in the top `top_share`.

    run_scores is (K runs x papers) in the same paper order as base_ranks.
    """
    limit = top_share * len(base_ranks)
    in_top = np.zeros(len(base_ranks))
    for run in run_scores:
        in_top += ranking.to_ranking(pd.Series(run)).to_numpy() <= limit
    return pd.Series(in_top / len(run_scores), index=base_ranks.index)
