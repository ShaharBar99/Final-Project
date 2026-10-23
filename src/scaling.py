"""Block B - Min-Max scaling, fitted once on the original graph and then frozen."""
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fit_scaler(features):
    """Learn the min and max of every feature column (original graph only)."""
    scaler = MinMaxScaler(clip=False)
    scaler.fit(features)
    return scaler


def apply_scaler(scaler, features):
    """Scale a feature table with an already fitted scaler.

    Nothing is clipped on purpose: on a perturbed graph a value can end up
    outside [0, 1], and that is exactly the damage we want to keep visible.
    """
    scaled = scaler.transform(features)
    return pd.DataFrame(scaled, index=features.index, columns=features.columns)
