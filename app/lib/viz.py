"""Small shared presentation helpers (Netflix-themed)."""
from __future__ import annotations

RISK_COLORS = {"High": "#E50914", "Medium": "#E0A800", "Low": "#2E9E5B"}
RISK_ORDER = ["Low", "Medium", "High"]
SEGMENT_ORDER = ["Low", "Medium", "High"]


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"
