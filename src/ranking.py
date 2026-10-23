"""Block D - turning scores into rankings."""
import numpy as np
import pandas as pd

from . import model

SIGNIFICANT_DIGITS = 12


def round_scores(scores, digits=SIGNIFICANT_DIGITS):
    """Round to `digits` significant digits.

    Scores that should be equal sometimes differ in the 16th digit (floating
    point noise from PageRank iterations or from scaling). Without rounding,
    such ties are broken at random and look like rank changes that are not
    really there. On our data the smallest real gap between two PageRank
    values is ~1e-8 (relative), the noise is ~1e-16, so 12 digits separates
    them safely.
    """
    values = scores.to_numpy(dtype=float)
    with np.errstate(divide="ignore"):
        magnitude = np.where(values == 0, 0, np.floor(np.log10(np.abs(values))))
    rounded = np.round(values / 10.0 ** magnitude, digits - 1) * 10.0 ** magnitude
    return pd.Series(rounded, index=scores.index)


def to_ranking(scores):
    """Rank 1 = most important. Papers with equal scores share the average rank.

    Averaging keeps ties from creating differences that are not in the data:
    if 500 papers have the same score, none of them is "better" than the
    others, so they all get the same middle rank.
    """
    return round_scores(scores).rank(ascending=False, method="average")


def base_scores(x_scaled, models):
    """Scores of every method on the original graph, one column per method.

    x_scaled : scaled feature table of the original graph
    models   : dict name -> trained forest
    The single-centrality baselines just use their own (scaled) column.
    """
    table = pd.DataFrame(index=x_scaled.index)
    for name, forest in models.items():
        table[f"rf_{name}"] = model.predict_scores(forest, x_scaled)
    for column in model.FEATURES_WITH_PAGERANK:
        table[column] = x_scaled[column]
    return table


def tie_share(scores):
    """Fraction of papers that share their score with at least one other paper."""
    return round_scores(scores).duplicated(keep=False).mean()
