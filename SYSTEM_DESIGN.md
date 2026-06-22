# Netflix Customer Retention Intelligence DSS — System Design

**Course:** AT82.04 Business Intelligence and Analytics (Intersem 2026)
**Team:** Natawat Damrongsilp (st125841), Liza Shrestha (st126674), Subhana Chitrakar (st126138)
**Pipeline:** ETL → Data Preparation → Model → Dashboard
**Status:** Design (build follows after approval)

This document is the blueprint for the whole system. It maps every part back to
the **proposal** and the **11-slide presentation**, and grounds each decision in
the **actual dataset** (`netflix_customer_churn.csv`, 5,000 rows).

---

## 1. Goal & framing

Build a **Decision Support System (DSS)** that helps a **Customer Retention
Manager** predict which subscribers will churn, understand the drivers, and
decide retention actions — *before* customers leave.

> **Honesty (keep in report):** the dataset is **synthetic** and **~50/50
> balanced** (2,515 churned / 2,485 retained). Real Netflix churn is far more
> imbalanced and not public. The project is a **methodology demonstration on a
> representative dataset**, not analysis of real Netflix customers.

---

## 2. Traceability — presentation → system

| Slide | Promise | Where it is built |
|---|---|---|
| 6 — Architecture (Data / Analytics / Output) | 3-layer system | §4 (ETL → Model → Dashboard map onto the 3 layers) |
| 7 — Descriptive Analytics + 4 KPIs | profiles, churn patterns, engagement | Dashboard *Intelligence* pages (§7) |
| 8 — Predictive Analytics (LR/DT/RF/XGBoost) + outputs | models + churn score, risk, rev-at-risk, drivers, metrics | Model stage (§6) |
| 9 — Customer Risk Table + retention by tier | per-customer table + playbook | Dashboard *Choice* page (§7) |
| 10 — Interactive Dashboard + Retention Strategy | unified BI + what-if simulation | Dashboard *Choice* page (§7) |
| 4 — Five Objectives | the five strategic objectives | covered across §5–§7 (mapped in §9) |

---

## 3. Dataset specification

`data/raw/netflix_customer_churn.csv` — 5,000 rows, 14 columns.

| Column | Role | Type | Notes |
|---|---|---|---|
| `customer_id` | key | id | drop from features; keep for the customer table |
| `age` | feature | numeric | |
| `gender` | feature | categorical | Male / Female / Other |
| `subscription_type` | feature | categorical | Basic / Standard / Premium |
| `watch_hours` | feature | numeric | total watch hours (engagement) |
| `last_login_days` | feature | numeric | recency — **leakage-audit candidate** (§5.5) |
| `region` | feature | categorical | Africa / Europe / Asia / Oceania / … |
| `device` | feature | categorical | TV / Mobile / Laptop / … |
| `monthly_fee` | feature + revenue | numeric | 8.99 / 13.99 / 17.99 → revenue basis |
| **`churned`** | **TARGET (y)** | binary | **1 = left, 0 = stayed** |
| `payment_method` | feature | categorical | Gift Card / Crypto / PayPal / Debit Card / … |
| `number_of_profiles` | feature | numeric | |
| `avg_watch_time_per_day` | feature | numeric | engagement |
| `favorite_genre` | feature | categorical | Action / Drama / … |

**⚠️ Gap to resolve — "Avg Subscription Duration":** slides 6 & 7 mention
*Duration* / *Avg Subscription Duration*, but the dataset has **no signup date or
tenure column**. Options:
- **(A, recommended)** replace that KPI with one we can compute, e.g. **Avg Watch
  Hours** or **Avg Monthly Fee**, and state the limitation; or
- **(B)** keep "Avg Subscription Duration" but mark it **N/A — not in dataset**; or
- **(C)** if the secondary dataset (abdulwadood) has tenure, source it there.

**Decision (locked): use Avg Monthly Fee** as the 4th KPI (computable; revenue
angle). The original *Avg Subscription Duration* is dropped and noted as a
limitation, so the dashboard never shows a fabricated number.

---

## 4. Architecture (maps to slide 6)

```
        DATA LAYER                 ANALYTICS LAYER              OUTPUT LAYER
 ┌────────────────────┐      ┌────────────────────────┐    ┌──────────────────┐
 │  ETL               │      │  Data Prep + Model     │    │  Dashboard       │
 │  Extract→Clean→Load│ ───▶ │  features → LR/DT/RF/   │──▶ │  Streamlit DSS   │
 │  → PostgreSQL      │      │  XGBoost → scores/tiers│    │  (Simon I·D·C)   │
 │  star schema+views │      │  → drivers, metrics    │    │                  │
 └────────────────────┘      └────────────────────────┘    └──────────────────┘
   raw CSV → clean tables       analysis-ready features        KPIs, risk table,
   + KPI views                  + trained model + scores        simulation
```

**Tech stack:** Python 3.11 · pandas · scikit-learn · XGBoost · PostgreSQL
(Docker) · Streamlit · Makefile · docker-compose.

> **On the dashboard reading the DB:** the BI layer (KPI views) lives in
> PostgreSQL. The Streamlit app reads KPI views from Postgres **when available**,
> and **falls back to the scored CSV** so the demo always runs without Docker.

---

## 5. Stage 1–2 — ETL & Data Preparation

### 5.1 Extract (`src/etl/extract.py`)
Read raw CSV; log row count + column list to the reproducibility log.

### 5.2 Transform / clean (`src/etl/transform.py`) — *cleaning is part of ETL*
- snake_case columns, enforce dtypes.
- Missing values: median (numeric) / "Unknown" (categorical); document per column.
- Drop duplicate `customer_id`.
- Range checks: `age` 0–120, `monthly_fee ≥ 0`, `watch_hours ≥ 0`,
  `last_login_days ≥ 0`; clip/flag outliers.
- Normalise categories (e.g. `subscription_type ∈ {Basic, Standard, Premium}`).
- Force `churned ∈ {0,1}`.

### 5.3 Load (`src/etl/load.py` + `sql/schema.sql`)
Star-ish schema + KPI views in PostgreSQL:

| Table / View | Purpose |
|---|---|
| `fact_subscriber` | one row per customer: features + `churned` + `monthly_fee` |
| `dim_plan` | plan → monthly_fee |
| `dim_region`, `dim_device`, `dim_genre` | lookups |
| `vw_kpi_summary` | total subscribers, total revenue, churn rate, revenue by plan |
| `vw_churn_by_segment` | churn rate by plan / region / engagement tier |

Also writes `data/processed/clean.parquet` as the canonical analysis table.

### 5.4 Data Preparation (`src/prep/`)
- **Target:** `churned`.
- **Numeric features:** age, watch_hours, last_login_days, monthly_fee,
  number_of_profiles, avg_watch_time_per_day.
- **Categorical:** gender, subscription_type, region, device, payment_method,
  favorite_genre → one-hot (ColumnTransformer).
- **Engineered:** `watch_per_profile = watch_hours / number_of_profiles`;
  `engagement_segment` (Low/Med/High via watch_hours + avg_watch_time_per_day
  tertiles); recency bucket from `last_login_days`.
- **Scaling:** standardize numerics (needed for LR; trees are scale-invariant) —
  inside the pipeline so it's leak-free.
- **Split:** stratified 80/20 on `churned`, fixed `random_state=42`.
- Persist the fitted preprocessing pipeline → `models/preprocess.pkl`.

### 5.5 Leakage audit (rigor point)
- `last_login_days` is the prime suspect: recency can be almost a proxy for
  churn. Plan: **train with vs. without it (and the recency bucket); report the
  PR-AUC / ROC-AUC delta** and discuss. No post-churn fields exist (no
  cancellation reason/date), so the rest are safe.

---

## 6. Stage 3 — Model (maps to slide 8)

### 6.1 Baselines (mandatory)
- **Majority class** (predict "stay") — on balanced data this ≈ 50% accuracy,
  which itself demonstrates why accuracy is weak here.
- **Heuristic:** `last_login_days ≥ T` → churn (tune T).
- **Logistic Regression** as the interpretable baseline.

### 6.2 Models
Logistic Regression → Decision Tree → Random Forest → **XGBoost** (simple →
complex, to *prove* the gain).

### 6.3 Evaluation (`src/model/evaluate.py`)
- **Stratified k-fold CV**, report **PR-AUC, ROC-AUC, F1, precision, recall
  (mean ± std)**. Accuracy reported too (valid here because data is balanced),
  but **not** the headline.
- **Calibration:** reliability curve + **Brier score** before turning
  probabilities into tiers.
- **Model selection:** best **PR-AUC** (tie-break ROC-AUC), then calibrate.
- Output `reports/metrics.json` + comparison table + calibration plot.

### 6.4 Risk scoring (`src/model/score.py`)
- Score every customer → calibrated `churn_probability`.
- **Risk tiers** (configurable): Low `<0.40`, Medium `0.40–0.70`, High `≥0.70`.
- **Revenue at risk** = `churn_probability × monthly_fee × 12` (annualised);
  monthly variant also stored. This is what makes prioritisation value-weighted.
- Output `data/processed/scored_customers.parquet` with: `customer_id,
  churn_probability, risk_tier, monthly_fee, revenue_at_risk, recommended_action,
  engagement_segment, key_factor*`.

### 6.5 Drivers — **feature importance, not SHAP** (per team decision)
- **Global "Key Churn Drivers":** model `feature_importances_` (RF/XGBoost) +
  LR coefficients → ranked bar chart in the dashboard.
- **Per-customer "Key Reason":** kept **illustrative**. Optional transparent
  heuristic (`key_factor*`): the top global-driver feature on which the customer
  sits in the adverse tail (e.g. very high `last_login_days`). Clearly labelled
  as a heuristic, **not** a causal claim. Easy to hide if not wanted.

---

## 7. Stage 4 — Dashboard (Simon: Intelligence → Design → Choice)

Streamlit multipage app. Maps slides 7 → 10.

### Home — KPI Overview *(Intelligence)*
KPI cards: **Total Subscribers**, **Total Revenue (MRR = Σ monthly_fee)**,
**Overall Churn Rate**, and **Avg Monthly Fee** (4th KPI — replaces *Avg
Subscription Duration*, which the dataset can't support). Plus churn-risk
distribution bar + revenue by plan.

### Page 1 — Descriptive Analytics *(Intelligence; slide 7)*
- Demographics: age distribution, gender, region.
- Subscription behaviour: plan mix, payment method, device, favourite genre.
- **Churned vs retained** comparison across features.
- **Engagement segmentation** (Low/Med/High) and its churn rate.

### Page 2 — Churn Risk *(Design; slide 8)*
- Predictive KPIs: **predicted churn rate**, **# high-risk customers**,
  **revenue at risk**, **retention opportunity score**.
- **Key churn drivers** (feature-importance chart).
- **Model performance** panel: PR-AUC / ROC-AUC / F1 / precision / recall +
  calibration curve, with the baseline comparison.

### Page 3 — Customer Table & Retention Simulation *(Choice; slides 9–10)*
- **Customer prediction table:** `customer_id, churn_probability, risk_tier,
  revenue_at_risk, recommended_action` (+ optional key_factor).
- **Filters:** customer segment, subscription plan, engagement level, churn
  risk, revenue value. *(Already promised in the slides.)*
- **Retention playbook** by tier (High/Medium/Low) — from slide 9.
- **What-if simulation** (slide 10):

  ```
  inputs:  target group (e.g. High risk), save_rate s, optional discount d
  expected_churners        = Σ churn_probability  (over group)
  baseline_rev_at_risk     = Σ churn_probability × monthly_fee × 12
  customers_saved          = expected_churners × s
  revenue_retained         = baseline_rev_at_risk × s
  campaign_cost (if disc.) = d × monthly_fee × 12 × group_size
  net_benefit              = revenue_retained − campaign_cost
  new_churn_rate           = (Σ churn_prob − customers_saved) / N
  ```
  Outputs mirror the slide: **Est. Churn Reduction, Revenue Retained,
  Customers Saved** (+ net benefit). Save rates are **assumptions** (state it).

---

## 8. Repository structure (target)

```
Netflix_Retention_DSS/
├── README.md
├── SYSTEM_DESIGN.md            ← this file
├── requirements.txt
├── Makefile                    # setup · etl · prep · train · score · dashboard · all
├── docker-compose.yml          # PostgreSQL 16
├── .env.example                # DB creds, paths
├── data/
│   ├── raw/netflix_customer_churn.csv
│   └── processed/              # clean.parquet, scored_customers.parquet
├── sql/
│   └── schema.sql              # star schema + KPI views
├── src/
│   ├── config.py               # paths, feature lists, tier thresholds, seed
│   ├── etl/      {extract,transform,load}.py
│   ├── prep/     {features,split}.py
│   └── model/    {baselines,train,evaluate,score}.py
├── app/
│   ├── Home.py
│   ├── pages/{1_Descriptive_Analytics,2_Churn_Risk,3_Retention_Simulation}.py
│   └── lib/{data_loader,inference,viz}.py
├── models/                     # preprocess.pkl, churn_model.pkl
└── reports/                    # metrics.json, figures/
```

---

## 9. Objectives coverage (slide 4)

| # | Objective | Delivered by |
|---|---|---|
| 1 | Customer characteristics & subscription behaviour | Descriptive page |
| 2 | Churn pattern (churned vs retained, engagement) | Descriptive page + engagement segmentation |
| 3 | Revenue impact & customer value | Revenue-at-risk, revenue by plan, value-weighted tiers |
| 4 | Predictive churn analysis (LR/DT/RF/XGBoost + drivers) | Model stage + Churn Risk page |
| 5 | Interactive dashboard (risk, segmentation, simulation) | Streamlit app (all pages) |

---

## 10. KPI definitions (formulas)

| KPI | Formula | Computable from dataset? |
|---|---|---|
| Total Subscribers | `COUNT(*)` | ✅ |
| Total Revenue (MRR) | `Σ monthly_fee` | ✅ |
| Overall Churn Rate | `mean(churned)` (~50.3%) | ✅ |
| Avg Monthly Fee *(replaces Avg Subscription Duration)* | `mean(monthly_fee)` | ✅ |
| Predicted Churn Rate | `mean(churn_probability)` or share predicted-churn | ✅ |
| # High-Risk Customers | `COUNT(risk_tier = High)` | ✅ |
| Revenue at Risk | `Σ churn_probability × monthly_fee × 12` | ✅ |
| Retention Opportunity Score | `Σ_high-risk revenue_at_risk × assumed_save_rate` | ✅ (assumption-based) |

---

## 11. Run plan (Makefile targets)

```
make setup      # venv + pip install -r requirements.txt
make db         # docker compose up postgres (optional)
make etl        # extract → clean → load (+ clean.parquet)
make prep       # features + stratified split + preprocess.pkl
make train      # baselines + LR/DT/RF/XGBoost + CV metrics + calibration
make score      # score all customers → scored_customers.parquet
make dashboard  # streamlit run app/Home.py
make all        # etl → prep → train → score
```

---

## 12. Limitations & threats to validity (for the report)

- **Synthetic, balanced data** → metrics are optimistic vs real, imbalanced churn.
- **No tenure** → "Avg Subscription Duration" KPI cannot be computed honestly (§3).
- **Single-period churn label** → no time dynamics / survival modelling.
- **Simulation save-rates are assumptions**, not measured effects — a real A/B
  test would be required to validate them.
- **"Key reason" is a heuristic / illustrative**, not a causal explanation.

---

## 13. Build order (next steps after this design is approved)

1. Infra — `requirements.txt`, `config.py`, `Makefile`, `docker-compose.yml`, `.env.example`
2. ETL — extract / transform / load + `sql/schema.sql`
3. Data prep — features / split
4. Model — baselines / train / evaluate / score
5. Dashboard — Home + 3 pages + lib
6. Verify end-to-end on the 5,000-row CSV; write `reports/metrics.json`
