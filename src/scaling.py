"""Block B - Min-Max scaling, fitted once on the original graph and then frozen."""


def fit_scaler(features):
    """Learn the min and max of every feature column (original graph only)."""
    raise NotImplementedError("step 3")


def apply_scaler(scaler, features):
    """Scale a feature table with an already fitted scaler. No clipping."""
    raise NotImplementedError("step 3")
