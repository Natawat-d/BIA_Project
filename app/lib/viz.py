"""Shared presentation helpers + Netflix dark theme (Plotly template + CSS)."""
from __future__ import annotations

import copy

import pandas as pd
import plotly.io as pio
import streamlit as st

NETFLIX_RED = "#E50914"
NETFLIX_BG = "#141414"
RISK_COLORS = {"High": "#E50914", "Medium": "#E0A800", "Low": "#2E9E5B"}
RISK_ORDER = ["Low", "Medium", "High"]
SEGMENT_ORDER = ["Low", "Medium", "High"]
RECENCY_ORDER = ["Active", "Lapsing", "Dormant"]

# churned vs retained
CHURN_COLORS = {"Retained": "#3A7CA5", "Churned": "#E50914"}
CHURN_SEQ = ["#3A7CA5", "#E50914"]  # 0 = retained, 1 = churned

# Multi-category palette (red-forward, Netflix-ish, still distinguishable).
NETFLIX_COLORWAY = ["#E50914", "#3A7CA5", "#E0A800", "#2E9E5B",
                    "#B0B0B0", "#831010", "#7B61FF"]

# --- Netflix-styled dark Plotly template, applied to every figure ------------
_nf = copy.deepcopy(pio.templates["plotly_dark"])
_nf.layout.paper_bgcolor = "rgba(0,0,0,0)"
_nf.layout.plot_bgcolor = "rgba(0,0,0,0)"
_nf.layout.font.update(color="#F5F5F5", family="Helvetica Neue, Arial, sans-serif")
_nf.layout.title.font.update(color="#FFFFFF")
_nf.layout.colorway = NETFLIX_COLORWAY
pio.templates["netflix"] = _nf
pio.templates.default = "netflix"


_THEME_CSS = """
<style>
[data-testid="stMetric"] {
    background: #1b1b1b;
    border: 1px solid #2b2b2b;
    border-left: 3px solid #E50914;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] { color: #FFFFFF; }
[data-testid="stMetricLabel"] p { color: #B3B3B3; }
h1, h2, h3 { letter-spacing: 0.2px; }
hr { border-color: #2b2b2b; }
</style>
"""


def inject_theme_css() -> None:
    """Netflix card styling for metrics; call once per page after set_page_config."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def add_status(df: pd.DataFrame) -> pd.DataFrame:
    """Add a readable 'Status' column from the 0/1 churn label."""
    df = df.copy()
    df["Status"] = df["churned"].map({0: "Retained", 1: "Churned"})
    return df
