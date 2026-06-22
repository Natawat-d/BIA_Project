"""Model — score every customer: churn probability, risk tier, revenue at risk.

Run with:  python -m src.model.score
Writes data/processed/scored_customers.csv (consumed by the dashboard).
"""
from __future__ import annotations

import logging

import pandas as pd
from joblib import load

from src.config import (
    ACTIONS,
    CLEAN_CSV,
    ID_COL,
    MODEL_PATH,
    REVENUE_MONTHS,
    RISK_HIGH_MIN,
    RISK_LOW_MAX,
    SCORED_CSV,
)
from src.prep.features import add_engineered, split_X_y

log = logging.getLogger(__name__)


def risk_tier(p: float) -> str:
    if p >= RISK_HIGH_MIN:
        return "High"
    if p <= RISK_LOW_MAX:
        return "Low"
    return "Medium"


def _key_factor(row, med) -> str:
    """Transparent, illustrative heuristic — NOT a causal claim."""
    if row["last_login_days"] >= med["last_login_days"] * 1.5:
        return "Low recent activity"
    if row["avg_watch_time_per_day"] <= med["avg_watch_time_per_day"] * 0.5:
        return "Low engagement"
    if row["monthly_fee"] >= med["monthly_fee"]:
        return "Higher plan cost"
    return "Stable usage"


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python -m src.model.train` first.")

    df = add_engineered(pd.read_csv(CLEAN_CSV))
    bundle = load(MODEL_PATH)
    model = bundle["model"]

    X, _, _, _ = split_X_y(df)
    proba = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({ID_COL: df[ID_COL], "churn_probability": proba})
    out["risk_tier"] = out["churn_probability"].apply(risk_tier)
    out["monthly_fee"] = df["monthly_fee"].values
    out["revenue_at_risk"] = out["churn_probability"] * out["monthly_fee"] * REVENUE_MONTHS
    out["recommended_action"] = out["risk_tier"].map(ACTIONS)

    med = df[["last_login_days", "avg_watch_time_per_day", "monthly_fee"]].median()
    out["key_factor"] = df.apply(lambda r: _key_factor(r, med), axis=1)

    # carry descriptive columns for dashboard filtering
    for col in ["subscription_type", "engagement_segment", "region", "age", "churned"]:
        out[col] = df[col].values

    out.to_csv(SCORED_CSV, index=False)
    high = int((out["risk_tier"] == "High").sum())
    total_rar = float(out["revenue_at_risk"].sum())
    log.info("Scored %d customers -> %s", len(out), SCORED_CSV)
    log.info("Mean churn prob=%.3f | High-risk=%d | Revenue at risk=$%s",
             float(proba.mean()), high, f"{total_rar:,.0f}")


if __name__ == "__main__":
    run()
