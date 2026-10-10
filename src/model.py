"""Block C - Random Forest that maps the feature vector of a node to an importance score."""


def train_model(x, y, seed=None):
    """Fit the forest on the original graph. y is the PageRank of every node."""
    raise NotImplementedError("step 4")


def predict_scores(model, x):
    """Score the nodes of any graph with the already trained model (no retraining)."""
    raise NotImplementedError("step 4")
