"""Dashboard data access. Reads the pipeline's CSV / JSON outputs (cached).

PostgreSQL is the BI layer of record, but the app reads the CSV outputs so the
demo always runs without Docker.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]      # Netflix_Retention_DSS/
sys.path.insert(0, str(ROOT))
from src.prep.features import add_engineered     # noqa: E402

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
CLEAN = PROCESSED / "clean.csv"
SCORED = PROCESSED / "scored_customers.csv"
METRICS = REPORTS / "metrics.json"


def has_clean() -> bool:
    return CLEAN.exists()


def has_scored() -> bool:
    return SCORED.exists()


@st.cache_data(show_spinner=False)
def load_clean() -> pd.DataFrame:
    """Clean table + engineered features (engagement segment, recency, etc.)."""
    return add_engineered(pd.read_csv(CLEAN))


@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame:
    return pd.read_csv(SCORED)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(METRICS.read_text()) if METRICS.exists() else {}
