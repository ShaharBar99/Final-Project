"""Block F - how much do scores and rankings move under perturbation."""


def node_instability(score_matrix):
    """Coefficient of variation (std / mean) per node; 0 when the mean is 0."""
    raise NotImplementedError("step 8")


def rank_correlations(base_scores, perturbed_scores):
    """Spearman and Kendall between the base ranking and one perturbed ranking."""
    raise NotImplementedError("step 8")
