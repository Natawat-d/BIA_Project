"""ETL — Extract: read the raw subscriber CSV."""
from __future__ import annotations

import logging

import pandas as pd

from src.config import RAW_CSV

log = logging.getLogger(__name__)


def extract() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_CSV}.\n"
            "Set NETFLIX_CSV=/path/to/netflix_customer_churn.csv "
            "or place the file in data/raw/."
        )
    df = pd.read_csv(RAW_CSV)
    log.info("Extracted %d rows x %d cols from %s", len(df), df.shape[1], RAW_CSV.name)
    return df
