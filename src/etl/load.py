"""ETL — Load: write the clean table to CSV and (optionally) PostgreSQL + views.

Run the whole ETL with:  python -m src.etl.load
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config import CLEAN_CSV, pg_url
from src.etl.extract import extract
from src.etl.transform import transform

log = logging.getLogger(__name__)

VW_KPI_SUMMARY = """
CREATE OR REPLACE VIEW vw_kpi_summary AS
SELECT COUNT(*) AS total_subscribers,
       SUM(monthly_fee) AS total_revenue_mrr,
       AVG(churned::numeric) AS overall_churn_rate,
       AVG(monthly_fee) AS avg_monthly_fee
FROM fact_subscriber;
"""

VW_REVENUE_BY_PLAN = """
CREATE OR REPLACE VIEW vw_revenue_by_plan AS
SELECT subscription_type, COUNT(*) AS subscribers,
       SUM(monthly_fee) AS revenue_mrr, AVG(churned::numeric) AS churn_rate
FROM fact_subscriber GROUP BY subscription_type ORDER BY revenue_mrr DESC;
"""

VW_CHURN_BY_SEGMENT = """
CREATE OR REPLACE VIEW vw_churn_by_segment AS
SELECT subscription_type, region, COUNT(*) AS subscribers,
       AVG(churned::numeric) AS churn_rate
FROM fact_subscriber GROUP BY subscription_type, region ORDER BY churn_rate DESC;
"""


def load_to_postgres(df: pd.DataFrame) -> bool:
    """Optional BI layer. Skips gracefully if PostgreSQL is not available."""
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(pg_url())
        with engine.begin() as conn:
            df.to_sql("fact_subscriber", conn, if_exists="replace", index=False)
            for ddl in (VW_KPI_SUMMARY, VW_REVENUE_BY_PLAN, VW_CHURN_BY_SEGMENT):
                conn.execute(text(ddl))
        log.info("Loaded fact_subscriber + KPI views into PostgreSQL")
        return True
    except Exception as exc:  # noqa: BLE001 - optional dependency / service
        log.warning("PostgreSQL load skipped (%s). CSV output is still available.", exc)
        return False


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = transform(extract())
    df.to_csv(CLEAN_CSV, index=False)
    log.info("Wrote clean data -> %s", CLEAN_CSV)
    load_to_postgres(df)


if __name__ == "__main__":
    run()
