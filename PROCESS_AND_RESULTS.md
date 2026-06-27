# Netflix Retention DSS — Process & Results

End-to-end record of **what each stage does** and **the actual results** from the
verified run on `netflix_customer_churn.csv` (5,000 subscribers).

- **Pipeline:** ETL → Data Preparation → Model → Dashboard
- **Run environment:** Python 3.14, pandas 3.0, scikit-learn 1.8 (XGBoost not
  installed during this run, so the comparison is **LR / Decision Tree / Random
  Forest**; XGBoost joins automatically after `make setup`).
- **Reproducibility:** fixed seed `random_state=42`, stratified 80/20 split,
  5-fold stratified CV.

> **Data honesty.** The dataset is **synthetic and ~50/50 balanced**
> (2,515 churned / 2,485 retained). This is a *methodology demonstration*; the
> very high scores below are a consequence of clean, separable synthetic data and
> would be lower and more imbalanced on real subscribers.

---

## 0. Dataset at a glance

| Property | Value |
|---|---|
| Rows | 5,000 |
| Target | `churned` (1 = left, 0 = stayed) |
| Overall churn rate | **50.3%** |
| Total revenue (MRR = Σ monthly_fee) | **$68,417** |
| Avg monthly fee | **$13.68** |
| Churn by plan | Basic **61.8%**, Standard 45.4%, Premium 43.7% |

**Read:** Basic-plan subscribers churn most (61.8%) — the cheapest tier is the
least sticky. Plans are roughly equal in price tiers ($8.99 / $13.99 / $17.99).

---

## 1. Stage 1 — ETL (Extract · Transform · Load)

**Process**
- **Extract** (`src/etl/extract.py`): read the raw CSV; log row/column counts.
- **Transform / clean** (`src/etl/transform.py`): lowercase columns; coerce
  numeric dtypes; drop duplicate `customer_id`; impute missing (median / "Unknown");
  range-check `age 0–120`, `monthly_fee ≥ 0`, `watch_hours ≥ 0`,
  `last_login_days ≥ 0`; force `churned ∈ {0,1}`.
- **Load** (`src/etl/load.py`): write `data/processed/clean.csv`; optionally load
  `fact_subscriber` + KPI views into PostgreSQL.

**Result**
```
Extracted 5000 rows x 14 cols
Clean dataset: 5000 rows; churn rate 0.503
Wrote clean data -> data/processed/clean.csv
PostgreSQL load skipped (no sqlalchemy) — CSV output still produced
```
- **0 rows dropped** (no duplicates / out-of-range values) — the synthetic data
  is already clean.
- PostgreSQL is optional; the run proceeded on CSV.

---

## 2. Stage 2 — Data Preparation

**Process** (`src/prep/`)
- **Target:** `churned`. **Dropped:** `customer_id` (identifier).
- **Engineered features:** `watch_per_profile`, `engagement_segment`
  (Low/Med/High from daily watch-time tertiles), `recency_bucket`
  (Active ≤7d / Lapsing ≤30d / Dormant >30d).
- **Encoding:** `StandardScaler` (numeric) + `OneHotEncoder` (categorical) inside
  a `ColumnTransformer` — fit only on training folds (leak-free).
- **Split:** stratified 80/20 (4,000 train / 1,000 test), `random_state=42`.

**Result:** model-ready feature matrix; preprocessing is bundled into each model
pipeline and persisted with the saved model.

---

## 3. Stage 3 — Model

**Process** (`src/model/`)
- **Baselines:** majority class + "inactive ≥ T days" heuristic (T tuned on train).
- **Models:** Logistic Regression → Decision Tree → Random Forest (→ XGBoost when
  installed), each as a `Pipeline(preprocessor, clf)`.
- **Validation:** 5-fold stratified CV reporting PR-AUC, ROC-AUC, F1, precision,
  recall, accuracy, Brier; **selection by CV PR-AUC**.
- **Calibration:** isotonic `CalibratedClassifierCV` on the best model.
- **Leakage audit:** retrain best model without recency features; report the delta.
- **Drivers:** model feature importance (no SHAP).

### 3.1 Model comparison (held-out test set)

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy | Brier |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.982 | 0.981 | 0.929 | 0.932 | 0.926 | 0.929 | 0.054 |
| Decision Tree | 0.978 | 0.982 | 0.953 | 0.977 | 0.930 | 0.954 | 0.037 |
| **Random Forest (best)** | **0.998** | **0.998** | **0.985** | **0.992** | **0.978** | **0.985** | **0.021** |

*(CV numbers match closely, e.g. RF CV PR-AUC 0.998 — low variance, no overf_)*

### 3.2 Baselines — the models clearly earn their place

| Baseline | Accuracy | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Majority class | 0.503 | 0.669 | 0.503 | 0.500 |
| Inactive ≥ 27 days | 0.716 | 0.731 | 0.708 | 0.759 |

**Read:** majority-class accuracy is **50.3%** — the headline reason we do **not**
lead on accuracy. Even the smarter "inactivity" rule (PR-AUC 0.708) is far below
Random Forest (0.998), so the ML adds real value.

### 3.3 Calibration (best model, isotonic)

| | PR-AUC | ROC-AUC | F1 | Accuracy | **Brier** |
|---|---|---|---|---|---|
| Calibrated Random Forest | 0.996 | 0.997 | 0.984 | 0.984 | **0.014** |

**Read:** Brier **0.014** (lower is better) → the probabilities are trustworthy,
which is required before turning them into risk tiers and revenue at risk.

### 3.4 Leakage audit (suspects: `last_login_days`, `recency_bucket`)

| Variant | PR-AUC | Accuracy | Brier |
|---|---|---|---|
| With recency features | 0.998 | 0.985 | 0.021 |
| Without recency features | 0.989 | 0.948 | 0.047 |
| **Δ (drop when removed)** | **0.009** | 0.037 | — |

**Read:** removing recency costs only **0.009 PR-AUC** — recency helps but the
model is **not** merely a recency proxy. No post-churn fields exist, so leakage
risk is low.

### 3.5 Key churn drivers (Random Forest feature importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | avg_watch_time_per_day | 0.194 |
| 2 | engagement_segment = High | 0.137 |
| 3 | watch_hours | 0.122 |
| 4 | last_login_days | 0.083 |
| 5 | engagement_segment = Low | 0.072 |
| 6 | watch_per_profile | 0.061 |
| 7 | number_of_profiles | 0.058 |
| 8 | recency_bucket = Dormant | 0.058 |

**Read:** churn is driven overwhelmingly by **engagement** (daily watch time,
total watch hours, engagement segment) and **recency** — not by demographics or
price. This directly supports the retention playbook (re-engage low-engagement
users).

---

## 4. Stage 4 — Scoring (per-customer output)

**Process** (`src/model/score.py`): score all 5,000 customers with the calibrated
model → churn probability → risk tier (Low ≤0.40, High ≥0.70, else Medium) →
`revenue_at_risk = churn_probability × monthly_fee × 12` → tier-based recommended
action + an illustrative `key_factor` heuristic. Output:
`data/processed/scored_customers.csv`.

**Result**

| Risk tier | Customers | Revenue at risk (annual) |
|---|---|---|
| Low | 2,491 | $4,149 |
| Medium | 3 | $245 |
| High | 2,506 | $393,285 |
| **Total** | **5,000** | **$397,679** |

- Mean predicted churn probability **0.504** ≈ actual churn rate 0.503 → globally
  well-calibrated.
- **Bimodal tiers** (almost everyone Low or High, only 3 Medium): the separable
  synthetic data makes the model very confident (probabilities near 0 or 1), so
  few land in the 0.40–0.70 band. On real data the Medium band would be populated.
- **99%** of revenue at risk sits in the High tier → prioritisation is sharp.

---

## 5. Dashboard (Streamlit · Simon: Intelligence → Design → Choice)

| Page | Phase | Contents |
|---|---|---|
| **Home** | Intelligence | 4 KPIs (Total Subscribers, Total Revenue MRR, Churn Rate, **Avg Monthly Fee**), churn-risk distribution, revenue by plan |
| **Descriptive Analytics** | Intelligence | demographics, plan/payment/device/genre mix, churned-vs-retained, engagement segmentation |
| **Churn Risk** | Design | predicted churn rate, # high-risk, revenue at risk, retention-opportunity score, driver chart, model-performance table + baselines + leakage note |
| **Retention Simulation** | Choice | filterable customer table (id, prob, tier, revenue at risk, key factor, action) + what-if campaign simulation |

**Simulation formula** (Retention Simulation page):
```
expected_churners = Σ churn_probability      (over target tier)
revenue_retained  = Σ revenue_at_risk × save_rate
customers_saved   = expected_churners × save_rate
campaign_cost     = discount × Σ(monthly_fee) × 12
net_benefit       = revenue_retained − campaign_cost
new_churn_rate    = (Σ churn_prob − customers_saved) / N
```
Save rates are **assumptions**, surfaced as such (a real A/B test would validate).

---

## 6. Findings (business reading)

1. **Engagement, not price, drives churn** — top drivers are watch time and
   recency. Retention spend should target low-engagement users.
2. **Basic plan churns most (61.8%)** despite being cheapest — low commitment.
3. **~$393k of annual revenue at risk sits in the High tier** — a small,
   well-defined group to act on first.
4. **ML beats the naive baselines decisively** (RF PR-AUC 0.998 vs 0.708) and is
   well-calibrated (Brier 0.014), so the risk scores are decision-grade.

---

## 7. Limitations & threats to validity

- **Synthetic, balanced, highly separable** → metrics are optimistic vs real,
  imbalanced churn; the Medium tier is nearly empty for the same reason.
- **No tenure/signup field** → "Avg Subscription Duration" replaced by Avg
  Monthly Fee.
- **Single-period churn label** → no time dynamics / survival analysis.
- **Drivers = feature importance** (global) and the per-customer "key factor" is a
  labelled heuristic — **not** causal explanations.
- **Simulation save-rates are assumptions**, not measured effects.
- **This run excluded XGBoost** (not installed); `make setup` adds it to the
  comparison.

---

## 8. How to reproduce

```bash
cd Netflix_Retention_DSS
make setup     # venv + install (adds XGBoost, Streamlit, Plotly)
make all       # etl -> train -> score   (writes clean.csv, metrics.json, scored_customers.csv)
make dashboard # launch the Streamlit DSS
```
Artifacts produced: `data/processed/clean.csv`, `data/processed/scored_customers.csv`,
`models/churn_model.pkl`, `reports/metrics.json`.
