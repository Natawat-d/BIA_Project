# Netflix Customer Retention Intelligence — DSS

A Decision Support System that predicts subscriber **churn**, scores **risk**,
quantifies **revenue at risk**, and **simulates retention strategies**.
Full blueprint: **[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)**.

**Pipeline:** ETL → Data Preparation → Model → Dashboard
**Stack:** Python · pandas · scikit-learn · XGBoost · PostgreSQL (optional) · Streamlit

## Quickstart
```bash
make setup        # create .venv and install dependencies
make etl          # clean + load -> data/processed/clean.csv (+ Postgres if running)
make train        # baselines + LR/DT/RF/XGBoost, CV metrics, calibration, leakage audit
make score        # per-customer churn score, risk tier, revenue at risk
make dashboard    # launch the Streamlit DSS
```
Run the whole pipeline at once: `make all` then `make dashboard`.

## Run with Docker (no local Python needed)
```bash
docker compose up --build
```
This starts **PostgreSQL**, runs the **pipeline** (ETL → train → score) once, then
serves the **dashboard** at **http://localhost:8501**. The raw CSV is mounted from
`../netflix_customer_churn.csv`; generated outputs are written back to the host.

```bash
docker compose run --rm pipeline   # re-run just the pipeline
docker compose down                # stop everything (add -v to wipe the DB)
```

> If you are not using the Makefile, activate the venv and run the modules
> directly: `python -m src.etl.load`, `python -m src.model.train`,
> `python -m src.model.score`, `streamlit run app/Home.py`.

## Data
Uses `netflix_customer_churn.csv` (5,000 synthetic subscribers; target = `churned`).
Location is auto-detected at `../netflix_customer_churn.csv`; override with
`NETFLIX_CSV=/path/to/file.csv`.

## Outputs
- `data/processed/clean.csv` — cleaned analysis table (ETL)
- `data/processed/scored_customers.csv` — per-customer churn score, tier, revenue at risk
- `reports/metrics.json` — model comparison, calibration, leakage audit, drivers
- `models/churn_model.pkl` — calibrated best model

## Notes
- Dataset is **synthetic and ~50/50 balanced** — a methodology demonstration, not real Netflix data.
- **PostgreSQL is optional** (BI views); the dashboard falls back to the CSV outputs.
- Churn **drivers use model feature importance** (no SHAP).
