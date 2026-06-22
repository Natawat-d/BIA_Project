"""Central configuration: paths, feature lists, thresholds, DB settings."""
from __future__ import annotations

import os
from pathlib import Path

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # Netflix_Retention_DSS/
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _resolve_raw_csv() -> Path:
    """Raw dataset: env override -> data/raw -> project parent (BIA_Project root)."""
    env = os.getenv("NETFLIX_CSV")
    if env:
        return Path(env)
    local = RAW_DIR / "netflix_customer_churn.csv"
    if local.exists():
        return local
    return ROOT.parent / "netflix_customer_churn.csv"


RAW_CSV = _resolve_raw_csv()
CLEAN_CSV = PROCESSED_DIR / "clean.csv"
SCORED_CSV = PROCESSED_DIR / "scored_customers.csv"
METRICS_JSON = REPORTS_DIR / "metrics.json"
MODEL_PATH = MODELS_DIR / "churn_model.pkl"

# --- modelling ---------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

ID_COL = "customer_id"
TARGET = "churned"

BASE_NUMERIC = [
    "age", "watch_hours", "last_login_days", "monthly_fee",
    "number_of_profiles", "avg_watch_time_per_day",
]
ENGINEERED_NUMERIC = ["watch_per_profile"]
BASE_CATEGORICAL = [
    "gender", "subscription_type", "region", "device",
    "payment_method", "favorite_genre",
]
ENGINEERED_CATEGORICAL = ["engagement_segment", "recency_bucket"]

NUMERIC_FEATURES = BASE_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = BASE_CATEGORICAL + ENGINEERED_CATEGORICAL

# Features that may leak (recency is almost a proxy for churn) — leakage audit.
LEAKAGE_SUSPECTS = ["last_login_days", "recency_bucket"]

# --- risk tiers & actions ----------------------------------------------------
RISK_LOW_MAX = 0.40      # prob <= 0.40 -> Low
RISK_HIGH_MIN = 0.70     # prob >= 0.70 -> High
REVENUE_MONTHS = 12      # annualise revenue at risk

ACTIONS = {
    "High": "Offer targeted discount / personalized content",
    "Medium": "Send re-engagement campaign",
    "Low": "Maintain engagement / loyalty rewards",
}

# --- optional PostgreSQL BI layer -------------------------------------------
PG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": os.getenv("PGPORT", "5432"),
    "db": os.getenv("PGDATABASE", "netflix_dss"),
    "user": os.getenv("PGUSER", "netflix"),
    "password": os.getenv("PGPASSWORD", "netflix"),
}


def pg_url() -> str:
    return (
        f"postgresql+psycopg2://{PG['user']}:{PG['password']}"
        f"@{PG['host']}:{PG['port']}/{PG['db']}"
    )
