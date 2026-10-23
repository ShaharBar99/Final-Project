"""Block D - turning scores into rankings."""
import pandas as pd

from . import model


def to_ranking(scores):
    """Rank 1 = most important. Papers with equal scores share the average rank.

    Averaging keeps ties from creating differences that are not in the data:
    if 500 papers have the same score, none of them is "better" than the
    others, so they all get the same middle rank.
    """
    return scores.rank(ascending=False, method="average")


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
    return scores.duplicated(keep=False).mean()
