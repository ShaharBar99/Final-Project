import numpy as np
import pandas as pd

from src import model


def toy_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.random((n, 4)), columns=["degree", "betweenness", "closeness", "kcore"])
    y = x["degree"] * 2 + x["closeness"]
    return x, y


def test_model_learns_a_simple_rule():
    x, y = toy_data()
    scores = model.holdout_check(x, y)
    assert scores["r2"] > 0.8


def test_same_seed_gives_same_scores():
    x, y = toy_data()
    a = model.predict_scores(model.train_model(x, y, seed=1), x)
    b = model.predict_scores(model.train_model(x, y, seed=1), x)
    assert np.allclose(a, b)


def test_prediction_uses_training_column_order():
    x, y = toy_data()
    m = model.train_model(x, y)
    shuffled = x[["kcore", "closeness", "degree", "betweenness"]]
    assert np.allclose(model.predict_scores(m, x), model.predict_scores(m, shuffled))


def test_model_does_not_retrain_on_new_data():
    x, y = toy_data()
    m = model.train_model(x, y)
    before = model.predict_scores(m, x)
    model.predict_scores(m, x * 5)  # scoring other data must not change the model
    assert np.allclose(before, model.predict_scores(m, x))
