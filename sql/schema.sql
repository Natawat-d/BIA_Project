-- Netflix Retention DSS — BI layer (PostgreSQL)
-- The ETL `load` step creates/populates fact_subscriber and (re)creates these
-- views. This file documents the intended schema and can also be run manually.

-- Fact table: one row per subscriber (populated by src/etl/load.py).
CREATE TABLE IF NOT EXISTS fact_subscriber (
    customer_id            TEXT PRIMARY KEY,
    age                    INTEGER,
    gender                 TEXT,
    subscription_type      TEXT,
    watch_hours            DOUBLE PRECISION,
    last_login_days        INTEGER,
    region                 TEXT,
    device                 TEXT,
    monthly_fee            DOUBLE PRECISION,
    churned                INTEGER,
    payment_method         TEXT,
    number_of_profiles     INTEGER,
    avg_watch_time_per_day DOUBLE PRECISION,
    favorite_genre         TEXT
);

-- KPI summary (Intelligence layer KPIs).
CREATE OR REPLACE VIEW vw_kpi_summary AS
SELECT
    COUNT(*)                              AS total_subscribers,
    SUM(monthly_fee)                      AS total_revenue_mrr,
    AVG(churned::numeric)                 AS overall_churn_rate,
    AVG(monthly_fee)                      AS avg_monthly_fee
FROM fact_subscriber;

-- Revenue by plan.
CREATE OR REPLACE VIEW vw_revenue_by_plan AS
SELECT subscription_type,
       COUNT(*)         AS subscribers,
       SUM(monthly_fee) AS revenue_mrr,
       AVG(churned::numeric) AS churn_rate
FROM fact_subscriber
GROUP BY subscription_type
ORDER BY revenue_mrr DESC;

-- Churn rate by segment (plan x region).
CREATE OR REPLACE VIEW vw_churn_by_segment AS
SELECT subscription_type,
       region,
       COUNT(*)              AS subscribers,
       AVG(churned::numeric) AS churn_rate
FROM fact_subscriber
GROUP BY subscription_type, region
ORDER BY churn_rate DESC;
