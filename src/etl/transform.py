"""ETL — Transform (clean). Cleaning is part of ETL, not a separate stage."""
from __future__ import annotations

import logging

import pandas as pd

from src.config import ID_COL, TARGET

log = logging.getLogger(__name__)

NUMERIC_COLS = [
    "age", "watch_hours", "last_login_days", "monthly_fee",
    "number_of_profiles", "avg_watch_time_per_day",
]
CATEGORICAL_COLS = [
    "gender", "subscription_type", "region", "device",
    "payment_method", "favorite_genre",
]


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # dtypes
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # duplicates
    before = len(df)
    df = df.drop_duplicates(subset=[ID_COL])
    if len(df) < before:
        log.info("Dropped %d duplicate %s rows", before - len(df), ID_COL)

    # missing values: median (numeric) / "Unknown" (categorical)
    for c in NUMERIC_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in CATEGORICAL_COLS:
        df[c] = (
            df[c].astype(str).str.strip()
            .replace({"nan": "Unknown", "": "Unknown", "None": "Unknown"})
        )

    # range validation
    mask = (
        df["age"].between(0, 120)
        & (df["monthly_fee"] >= 0)
        & (df["watch_hours"] >= 0)
        & (df["last_login_days"] >= 0)
        & (df["number_of_profiles"] >= 0)
    )
    dropped = (~mask).sum()
    if dropped:
        log.info("Dropped %d rows failing range checks", int(dropped))
    df = df[mask]

    # normalise target to {0, 1}
    df[TARGET] = (
        pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int).clip(0, 1)
    )

    df = df.reset_index(drop=True)
    log.info("Clean dataset: %d rows; churn rate %.3f", len(df), df[TARGET].mean())
    return df
