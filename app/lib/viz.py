"""Small shared presentation helpers (Netflix-themed)."""
from __future__ import annotations

import pandas as pd

NETFLIX_RED = "#E50914"
RISK_COLORS = {"High": "#E50914", "Medium": "#E0A800", "Low": "#2E9E5B"}
RISK_ORDER = ["Low", "Medium", "High"]
SEGMENT_ORDER = ["Low", "Medium", "High"]
RECENCY_ORDER = ["Active", "Lapsing", "Dormant"]

# churned vs retained
CHURN_COLORS = {"Retained": "#3A7CA5", "Churned": "#E50914"}
CHURN_SEQ = ["#3A7CA5", "#E50914"]  # 0 = retained, 1 = churned


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def add_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add a readable 'Status' column from the 0/1 churn label."""
    df = df.copy()
    df["Status"] = df["churned"].map({0: "Retained", 1: "Churned"})
    return df
