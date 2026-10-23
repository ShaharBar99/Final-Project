"""Block C - Random Forest that maps the feature vector of a node to an importance score."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from . import config

TARGET = "pagerank"

# the two versions of the model (decision D2 in docs/decisions.md)
FEATURES_WITH_PAGERANK = ["degree", "betweenness", "closeness", "pagerank", "kcore"]
FEATURES_WITHOUT_PAGERANK = ["degree", "betweenness", "closeness", "kcore"]

FOREST_SETTINGS = dict(n_estimators=200, min_samples_leaf=5, n_jobs=-1)


def train_model(x, y, seed=config.SEED):
    """Fit the forest on the original graph. y is the PageRank of every node.

    The model keeps the column order it was trained with, so predict_scores
    can check that it gets the same columns later.
    """
    model = RandomForestRegressor(random_state=seed, **FOREST_SETTINGS)
    model.fit(x, y)
    return model


def predict_scores(model, x):
    """Score the nodes of any graph with the already trained model (no retraining)."""
    columns = list(model.feature_names_in_)
    return pd.Series(model.predict(x[columns]), index=x.index)


def holdout_check(x, y, seed=config.SEED):
    """Sanity check only: train on 80 % of the papers, test on the other 20 %.

    The stability experiments use the model trained on all nodes (as in
    Phase A); this just tells us whether the forest learned anything real.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    model = train_model(x_train, y_train, seed)
    pred = predict_scores(model, x_test)
    return {
        "r2": r2_score(y_test, pred),
        "spearman": pred.corr(y_test, method="spearman"),
    }


def load_or_train(name, x, y, models_dir):
    """Load the saved forest `name`, or rebuild it (fixed seed) if the file is missing."""
    import joblib
    path = models_dir / f"rf_{name}.joblib"
    if path.exists():
        return joblib.load(path)
    forest = train_model(x, y)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, path)
    return forest
