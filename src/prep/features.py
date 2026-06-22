"""Data Preparation — feature engineering, encoding, X/y split.

Feature engineering happens here (after the data is cleaned in ETL).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features used by both modelling and the dashboard."""
    df = df.copy()

    # watch hours per profile (avoid divide-by-zero)
    profiles = df["number_of_profiles"].replace(0, np.nan)
    df["watch_per_profile"] = (df["watch_hours"] / profiles).fillna(df["watch_hours"])

    # engagement segment from daily watch time (tertiles)
    try:
        df["engagement_segment"] = pd.qcut(
            df["avg_watch_time_per_day"], q=3, labels=["Low", "Medium", "High"]
        ).astype(str)
    except ValueError:  # not enough distinct values
        df["engagement_segment"] = "Medium"

    # recency bucket from last login
    df["recency_bucket"] = pd.cut(
        df["last_login_days"],
        bins=[-1, 7, 30, np.inf],
        labels=["Active", "Lapsing", "Dormant"],
    ).astype(str)

    return df


def split_X_y(df: pd.DataFrame, drop_features: list[str] | None = None):
    """Return (X, y, numeric_cols, categorical_cols), optionally dropping features."""
    drop_features = drop_features or []
    num = [c for c in NUMERIC_FEATURES if c not in drop_features]
    cat = [c for c in CATEGORICAL_FEATURES if c not in drop_features]
    X = df[num + cat].copy()
    y = df[TARGET].astype(int).copy()
    return X, y, num, cat


def build_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    """Scale numerics (for LR) + one-hot categoricals. Leak-free inside a Pipeline."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )
