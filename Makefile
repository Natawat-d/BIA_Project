.PHONY: setup db etl train score dashboard all clean
PY ?= python

setup:               ## create venv + install dependencies
	$(PY) -m venv .venv && ./.venv/bin/pip install -U pip && ./.venv/bin/pip install -r requirements.txt

db:                  ## start optional PostgreSQL BI layer
	docker compose up -d postgres

etl:                 ## extract -> clean -> load (clean.csv + Postgres if available)
	$(PY) -m src.etl.load

train:               ## baselines + LR/DT/RF/XGBoost, CV, calibration, leakage audit
	$(PY) -m src.model.train

score:               ## score every customer -> scored_customers.csv
	$(PY) -m src.model.score

dashboard:           ## launch the Streamlit DSS
	streamlit run app/Home.py

all: etl train score ## run the full pipeline

clean:               ## remove generated artifacts
	rm -f data/processed/*.csv models/*.pkl reports/*.json
