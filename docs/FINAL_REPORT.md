NETFLIX CUSTOMER RETENTION INTELLIGENCE DECISION SUPPORT SYSTEM: A DATA-DRIVEN CHURN PREDICTION AND RETENTION PRIORITIZER FOR SUBSCRIPTION STREAMING

by

Group 6

Mr. Natawat Damrongsilp — st125841
Ms. Liza Shrestha — st126674
Ms. Subhana Chitrakar — st126138

A Group Project Report Submitted in Partial Fulfillment of the Requirements for
AT82.04 Business Intelligence and Analytics

Presented to:

Dr. Vatcharaporn Esichaikul

Asian Institute of Technology
School of Engineering and Technology
Thailand
July 2026

---

# ACKNOWLEDGMENTS

We are grateful to our course instructor, Dr. Vatcharaporn Esichaikul, for her guidance throughout this project. Her insistence on methodological precision — how exactly a churn probability is produced, and whether engineered features genuinely earn their place in a model — shaped the probability-methodology write-up and the validity checks that became central contributions of this report.

We thank the faculty of the School of Engineering and Technology at the Asian Institute of Technology for the coursework in AT82.04 Business Intelligence and Analytics that seeded this project, and our classmates for the discussions that kept the Customer Retention Manager's perspective at the centre of every design decision.

We are indebted to the publishers of the Netflix Customer Churn dataset on Kaggle for making a representative subscriber dataset publicly available. We also thank the open-source communities behind scikit-learn (Pedregosa et al., 2011), XGBoost (Chen & Guestrin, 2016), Streamlit, Plotly, pandas, PostgreSQL, and Docker for the tools on which this work is built.

Finally, we thank our families for their support during the weeks this project required.

# ABSTRACT

Subscription streaming platforms depend on recurring monthly fees, which makes customer churn a direct and compounding threat to revenue, yet retention teams typically lack tools that tell them who to save first and how much is at stake. This project presents the design, implementation, and evaluation of a fully deployed decision support system (DSS) for a Netflix-style streaming service that answers four operational questions: what characterises the subscriber base, which factors drive churn and how much revenue they endanger, whether individual churn risk can be predicted reliably enough to rank customers for retention spending, and how predictions should be delivered as decisions.

The system is a reproducible ETL → Data Preparation → Model → Dashboard pipeline (fixed seed 42, one-command rebuild) over the 5,000-subscriber, 14-column Netflix Customer Churn dataset (Kaggle; churn rate 50.3%, MRR $68,417). A PostgreSQL warehouse (fact table plus three KPI views) supports the descriptive layer; four classifiers — Logistic Regression, Decision Tree, Random Forest, and XGBoost — are trained inside leak-free pipelines with stratified cross-validation, naive baselines, and isotonic calibration. The CV-selected XGBoost attains held-out PR-AUC 1.000 (calibrated 0.9997, Brier 0.0054) against baselines of 0.503 and 0.708. Because a near-perfect score is a red flag rather than a triumph, the report contributes two validity diagnostics: an overfitting check (train-test PR-AUC gaps of +0.0001 to +0.016 across all four models; even a ~35-parameter linear model scores 0.982, so the results cannot be memorization) and a feature-family ablation (removing engagement features collapses PR-AUC to 0.796; demographics alone reach 0.578, barely above random 0.503), which localizes all predictive signal in behavior while disclosing that a snapshot dataset cannot distinguish early-warning signal from post-churn artifact — the reported scores are therefore an upper bound.

Scoring all 5,000 customers yields 2,507 high-risk subscribers and ≈$397,289 of annual revenue at risk; under a transparent accounting formula (Σ revenue at risk × assumed save rate) a 30% save rate on the high-risk tier corresponds to ≈$118,000/year of retention opportunity ($39k/$79k/$158k at 10%/20%/40%). The four-page Streamlit dashboard — Home, Descriptive Analytics, Churn Risk, and Customer Simulation — maps onto Simon's Intelligence–Design–Choice framework and runs live on AWS EC2 via Docker Compose, including a customer-level what-if simulator driven by the trained calibrated model. The contribution of the project is threefold: a reproducible open-data churn-DSS pipeline, an explicit account of how a calibrated churn probability is computed (log-odds → sigmoid, verified exact → isotonic calibration), and a validity-diagnostic framework that protects against over-selling near-perfect scores on synthetic, highly separable data.

Keywords: decision support system, customer churn, retention, XGBoost, calibration, PR-AUC, Netflix, subscription streaming

# CONTENTS

1. Introduction (1.1 Background of the Study · 1.2 Statement of the Problem · 1.3 Research Questions · 1.4 Objectives of the Study · 1.5 Scope and Limitations · 1.6 Organization of the Study)
2. Literature Review (2.1 Churn Prediction as Supervised Classification · 2.2 Probability Outputs and Calibration · 2.3 Evaluation Metrics for Churn Models · 2.4 Decision Support Systems and Data Warehousing · 2.5 Chapter Summary)
3. Methodology (3.1 Data Sources and ETL Pipeline · 3.2 Feature Engineering · 3.3 Churn Classification Models · 3.4 Churn-Probability Computation and Calibration · 3.5 Naive Baselines and Validity Checks · 3.6 Application Architecture · 3.7 Business Impact Calculation · 3.8 Chapter Summary)
4. Results and Discussion (4.1 Descriptive Analytics of the Subscriber Base · 4.2 Model Performance · 4.3 Validity Diagnostics: Overfitting Check and Feature-Family Ablation · 4.4 Business Impact · 4.5 Page-by-Page Walkthrough · 4.6 Discussion · 4.7 Chapter Summary)
5. Conclusion and Recommendations (5.1 Conclusions · 5.2 Recommendations · 5.3 Future Research Directions)
6. References

# LIST OF TABLES

Table 3.1 — Engineered feature catalogue (3 engineered features; 15 model features total).
Table 3.2 — Model portfolio and hyperparameters.
Table 3.3 — Technology stack.
Table 4.1 — Held-out test performance of the four models versus naive baselines.
Table 4.2 — Overfitting check: train vs. cross-validation vs. held-out-test PR-AUC.
Table 4.3 — Feature-family ablation (XGBoost test PR-AUC).
Table 4.4 — Retention-opportunity sensitivity to the save-rate assumption.
Table 4.5 — Example rows from the delivered customer prediction table.

# 1. Introduction

## 1.1 Background of the Study

Global video streaming has grown enormously, with Netflix among the leading platforms worldwide. As binge-watching became mainstream, the number of competing streaming services increased sharply, making long-term profitability highly competitive. Unlike traditional businesses, streaming platforms rely on a subscription revenue model, which makes customer churn — the cancellation of a subscription — a direct and compounding threat to revenue: every cancellation removes its monthly fee from Monthly Recurring Revenue (MRR) in every subsequent month. Because retaining an existing customer is substantially cheaper than acquiring a new one, the ability to identify *which* customers are likely to cancel — early enough to intervene — is a core business-intelligence capability for any subscription business.

This project designs, implements, and evaluates a complete **Decision Support System (DSS)** for a Netflix-style streaming service: the **Netflix Customer Retention Intelligence DSS**. The system integrates descriptive analytics (who churns, and what it costs) with predictive analytics (who *will* churn, with what probability, and what to do about it), delivered through an interactive dashboard deployed live on AWS. Unlike the proposal stage of this work, the system described here is fully built: a reproducible ETL → Data Preparation → Model → Dashboard pipeline, containerised with Docker and running in the cloud.

## 1.2 Statement of the Problem

Retention teams at subscription businesses face three practical gaps. First, churn is usually reported as a single aggregate rate, which hides *where* it concentrates — by plan, engagement level, or recency of use. Second, without per-customer risk estimates, retention budgets are spread evenly rather than aimed at the customers who are both likely to leave and expensive to lose. Third, the tools that do produce risk scores rarely explain how the score is computed or whether it can be trusted, which undermines adoption by non-technical decision makers.

The problem this project addresses is therefore twofold: *which subscribers are likely to cancel, and what proactive, prioritised actions can retain them* — answered by a system whose methodology is transparent enough to be audited, including an honest account of what its near-perfect scores do and do not mean.

## 1.3 Research Questions

The project is organised around four questions:

1. **RQ1 (Describe).** What characterises the subscriber base, and how do churned customers differ from retained ones across plan, engagement, recency, demographics, and revenue?
2. **RQ2 (Quantify).** Which factors drive churn, and how much revenue do they place at risk?
3. **RQ3 (Predict).** Can machine-learning models estimate each customer's churn probability reliably enough to rank customers for retention spending — and are near-perfect scores on this dataset trustworthy evidence of that ability?
4. **RQ4 (Decide).** How should predictions be delivered to a Customer Retention Manager as decisions — risk tiers, recommended actions, and what-if simulation — following Simon's Intelligence–Design–Choice framework?

## 1.4 Objectives of the Study

1. Analyze customer characteristics and subscription behavior.
2. Identify churn patterns and engagement trends (churned versus retained; low/medium/high engagement).
3. Assess revenue impact and customer value.
4. Conduct predictive churn analysis with machine-learning models, identify the key churn drivers, and evaluate the models rigorously.
5. Deploy an interactive retention-intelligence dashboard with risk scoring, segmentation, and customer-level what-if simulation.

All five objectives were achieved; Chapter 4 reports the delivered results and Chapter 5 maps them back to these objectives.

## 1.5 Scope and Limitations

The working data is the *Netflix Customer Churn* dataset (Kaggle): 5,000 subscriber records × 14 columns, a single snapshot with **no time dimension** (no signup date, tenure, or event timestamps). Netflix does not release real subscriber data; this dataset is synthetic and approximately balanced (churn rate 50.3%). The project is therefore framed as a **methodology demonstration on representative data**: the pipeline, validation protocol, and DSS design transfer to real data, but the near-perfect model scores of Chapter 4 would not — real churn is rarer, noisier, and harder to separate. This disclosure is treated as a first-class finding rather than a footnote, and Section 4.6.3 develops it into explicit threats to validity. The system is validated retrospectively on held-out data rather than via a live A/B pilot; campaign save rates are scenario assumptions.

## 1.6 Organization of the Study

Chapter 2 reviews the literature on churn prediction, probability calibration, evaluation metrics, and DSS design. Chapter 3 describes the methodology: data and ETL, feature engineering, the model portfolio, the churn-probability computation, the validity-check design, the application architecture, and the business-impact accounting. Chapter 4 presents results — descriptive findings, model performance, validity diagnostics, business impact, and a page-by-page walkthrough of the deployed application — followed by discussion. Chapter 5 concludes, states recommendations for practitioners and for similar student projects, and outlines future research.

# 2. Literature Review

## 2.1 Churn Prediction as Supervised Classification

Predicting customer defection from historical behavioral attributes is a long-established application of supervised machine learning. Neslin et al. (2006) formalised the problem in their churn-modeling "tournament" and showed that model choice materially affects the economic value of a retention campaign. Lemmens and Croux (2006) demonstrated that ensemble tree methods (bagging and boosting) significantly outperform single classifiers for churn. Verbeke et al. (2012) extended this line with profit-driven churn modeling, arguing that churn models should be judged by business value rather than raw accuracy — a principle this system adopts through revenue-at-risk weighting. Comparative studies consistently find gradient-boosted trees among the strongest performers: Vafeiadis et al. (2015) benchmarked standard classifiers on telecom churn and found boosted ensembles superior, and Ahmad, Jafar, and Aljoumaa (2019) reported XGBoost as the best model on a large real telecom dataset. Our model portfolio — Logistic Regression, Decision Tree, Random Forest (Breiman, 2001), and XGBoost (Chen & Guestrin, 2016) — mirrors this literature: an interpretable linear baseline, a single tree, and two ensembles.

## 2.2 Probability Outputs and Calibration

Churn intervention requires a *ranked* list (whom to contact first) and *dollar weighting* (probability × revenue), so the model must output a probability, not a 0/1 label. Classifiers produce probabilities by mapping an unbounded raw score (log-odds) through the logistic (sigmoid) function; however, these raw probabilities are often poorly calibrated, and boosted trees in particular tend to be over-confident (Niculescu-Mizil & Caruana, 2005). Two standard post-hoc corrections exist: Platt scaling, which fits a sigmoid to the scores (Platt, 1999), and isotonic regression, a non-parametric monotonic remapping (Zadrozny & Elkan, 2002). We use **isotonic calibration**, following Niculescu-Mizil and Caruana's (2005) finding that it is the more effective correction for tree ensembles when sufficient data is available.

## 2.3 Evaluation Metrics for Churn Models

Where the positive class is the business focus, the precision–recall curve is more informative than the ROC curve, particularly under class imbalance (Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015). Although our dataset happens to be balanced, real churn populations are imbalanced (typically 2–10% monthly), so we lead with **PR-AUC** for methodological transferability and report ROC-AUC, F1, precision, recall, accuracy, and the Brier score for completeness. Naive baselines — majority class and a single-rule heuristic — anchor every comparison, following the reporting standard that a model must demonstrably out-earn the obvious alternative.

## 2.4 Decision Support Systems and Data Warehousing

The system design follows Simon's (1960) three-phase model of decision making — Intelligence (find and understand the problem), Design (develop and analyse alternatives), and Choice (select and commit to action) — which maps naturally onto descriptive analytics, predictive modeling, and prescriptive recommendation/simulation. The storage layer follows the data-warehouse tradition of a subject-oriented, integrated, non-volatile analytical store (Inmon, 2005), organised as a fact table with analytical views in the dimensional-modeling style of Kimball and Ross (2013).

## 2.5 Chapter Summary

The literature supports four design commitments carried through this project: an ensemble-centred model portfolio with an interpretable baseline (2.1); probability-first outputs with isotonic calibration (2.2); PR-AUC-led evaluation against naive baselines (2.3); and a Simon-framework DSS over a warehouse-backed descriptive layer (2.4).

# 3. Methodology

## 3.1 Data Sources and ETL Pipeline

**Data source.** The *Netflix Customer Churn* dataset (Kaggle): 5,000 subscriber records × 14 columns, binary target `churned` (1 = cancelled). Numeric attributes: `age`, `watch_hours`, `last_login_days`, `monthly_fee`, `number_of_profiles`, `avg_watch_time_per_day`. Categorical: `gender`, `subscription_type` (Basic/Standard/Premium), `region` (six continents), `device`, `payment_method`, `favorite_genre`. Class balance ≈50/50 (churn rate 50.3%); total MRR $68,417; average monthly fee $13.68; no time dimension.

**Extract** reads the raw CSV (5,000 × 14) with a fail-fast existence check and logs row/column counts for verification.

**Transform (data cleaning)** applies, in order: (1) standardise column names (trim, lower-case); (2) coerce the six numeric columns to numeric dtype, converting invalid entries to missing; (3) drop duplicate `customer_id` rows; (4) impute missing values — median for numerics, "Unknown" for categoricals; (5) validate ranges (age 0–120; fee, watch hours, login recency, and profile counts ≥ 0) and drop violating rows; (6) normalise the target strictly to {0, 1}. On this dataset the cleaning pass loses zero rows.

**Load** writes two destinations: `data/processed/clean.csv` (consumed by modeling and the dashboard) and a **PostgreSQL data warehouse** — a denormalised fact table `fact_subscriber` (one row per subscriber) plus three KPI views (`vw_kpi_summary`, `vw_revenue_by_plan`, `vw_churn_by_segment`). The store is subject-oriented, integrated, and non-volatile (Inmon, 2005), with two disclosed simplifications: it is a single snapshot (not time-variant, since the data has no time dimension) and it uses one wide fact table rather than a full star schema (Kimball & Ross, 2013). The dashboard reads the processed CSVs directly and falls back gracefully if the database is offline, so the demonstration always runs.

## 3.2 Feature Engineering

The preparation stage adds three engineered features the raw data does not contain, giving 15 model features (7 numeric + 8 categorical). Table 3.1 lists them.

Table 3.1 — Engineered feature catalogue.

| Engineered feature | Construction | Purpose |
|---|---|---|
| `watch_per_profile` | `watch_hours` ÷ `number_of_profiles` (safe against division by zero) | Normalises viewing by household size; distinguishes one heavy viewer from a large shared account. |
| `engagement_segment` | Population tertiles of `avg_watch_time_per_day` → Low / Medium / High | Converts a skewed continuous signal into a stable behavioral tier; matches retention-team language; robust split for tree models. |
| `recency_bucket` | Fixed bands on `last_login_days`: Active (≤ 7), Lapsing (8–30), Dormant (> 30) | Encodes the recency dimension of RFM analysis as actionable, human-readable bands. |

These features are demonstrably *used*, on four independent grounds: (1) they are model inputs, one-hot encoded alongside the raw features; (2) they dominate the trained model's feature importance — `engagement_segment_High` (0.498) and `engagement_segment_Low` (0.243) are the top two drivers, together ≈74% of total importance, and `recency_bucket_Dormant` ranks fourth; (3) they structure the descriptive findings and are sidebar filters on every dashboard page; and (4) `recency_bucket` is deliberately removed in the leakage audit (Section 3.5) as a stress test.

All preprocessing — StandardScaler for numerics, one-hot encoding for categoricals, expanding 15 features to ≈35 model inputs — is wrapped inside a scikit-learn `Pipeline`, fit on training folds only, so no test information leaks. The data is split 80:20 with stratification to preserve the churn distribution.

## 3.3 Churn Classification Models

Four classifiers are trained inside the leak-free pipeline (Table 3.2), mirroring the literature's recommended portfolio: an interpretable linear baseline, a single tree, and two ensembles.

Table 3.2 — Model portfolio and hyperparameters.

| Model | Key hyperparameters | Role |
|---|---|---|
| Logistic Regression | max_iter 1000, class-balanced | Interpretable linear baseline (~35 parameters) |
| Decision Tree | max depth 6 | Single interpretable tree |
| Random Forest | 300 trees | Bagged ensemble (Breiman, 2001) |
| XGBoost | 400 trees, depth 5, learning rate 0.05, subsample 0.9 | Gradient-boosted ensemble (Chen & Guestrin, 2016) |

Model selection uses **stratified k-fold cross-validation on the training set only** (selection metric: PR-AUC); the held-out test set (20%, stratified) is scored once for honest reporting. The training run persists its artifacts — the calibrated model (`churn_model.pkl`) and the complete metric set (`metrics.json`) — and the dashboard reads both, so reported and displayed numbers share a single source.

## 3.4 Churn-Probability Computation and Calibration

This is the core of the predictive system, stated precisely:

1. **Raw score (log-odds).** Each classifier outputs an unbounded real-valued score. For Logistic Regression this is the linear combination *z = w·x + b*; for XGBoost it is the *margin* — a base value (initialised at the population churn rate, expressed in log-odds) plus the sum of all 400 boosted trees' leaf scores, each tree fit to correct the residual errors of its predecessors (Chen & Guestrin, 2016). We verified empirically that the margin is exactly additive and ranges roughly −15 to +17 on our data.
2. **Sigmoid link.** The logistic function *p = 1 / (1 + e^(−z))* maps the raw score to (0, 1). We verified that `sigmoid(raw score)` reproduces the library's `predict_proba` output exactly — maximum difference 0.0 across all 5,000 customers, for both Logistic Regression and XGBoost.
3. **Isotonic calibration.** Because boosted trees are characteristically over-confident (Niculescu-Mizil & Caruana, 2005), the selected model is wrapped in `CalibratedClassifierCV(method="isotonic", cv=5)` (Zadrozny & Elkan, 2002) — preferred over Platt scaling (Platt, 1999) for tree ensembles — so that a predicted 30% corresponds to an empirical churn rate of ≈30%. Calibration quality is measured by the Brier score (0.005 on the held-out test set).
4. **Business outputs.** The calibrated probability is converted into decision-ready fields per customer: `risk_tier` (Low ≤ 0.40; High ≥ 0.70; Medium otherwise), `revenue_at_risk = p × monthly_fee × 12`, and a tier-mapped `recommended_action`.

**Why a probability rather than a 0/1 label?** A hard label answers only "will they churn?"; retention management needs "who first, and how much is at stake?" A calibrated probability supports ranking customers under limited campaign budgets, revenue weighting (p × fee), and transparent, adjustable tier thresholds — none of which a binary label provides (Neslin et al., 2006; Verbeke et al., 2012).

## 3.5 Naive Baselines and Validity Checks

Three safeguards are designed into the evaluation protocol.

**Naive baselines.** Two baselines anchor all results: the majority class (PR-AUC 0.503) and the single rule "inactive ≥ 27 days" (PR-AUC 0.708). Any model must demonstrably out-earn both.

**Leakage audit.** Because recency nearly proxies churn (a churned user stops logging in), the selected model is retrained *without* `last_login_days` and `recency_bucket`, and the PR-AUC delta is reported.

**Validity diagnostics.** Two further checks respond to the near-perfect headline scores: an **overfitting check** comparing train, cross-validation, and held-out-test PR-AUC for every model (an overfit model shows a large train–test gap), and a **feature-family ablation** retraining XGBoost with whole feature families removed to localize the signal and bound the leakage risk. Results appear in Section 4.3.

## 3.6 Application Architecture

The system is one reproducible pipeline — ETL → Data Preparation → Model → Dashboard — containerised with Docker Compose as three services (`pipeline`, `postgres`, `dashboard`) and deployed on an AWS EC2 instance (region ap-southeast-7, Bangkok). Every run is deterministic (fixed random seed 42); one command (`docker compose up`) reproduces the entire system. Table 3.3 summarises the stack.

Table 3.3 — Technology stack.

| Layer | Technology | Role |
|---|---|---|
| Presentation | Streamlit + Plotly | Interactive multi-page dashboard |
| Application | Python 3.11 | Page scripts, data access, chart builders, caching |
| Analytics / ML | pandas, scikit-learn, XGBoost, joblib | ETL, feature engineering, training, calibration, scoring |
| Data & storage | PostgreSQL, CSV | Data warehouse (fact table + KPI views); processed pipeline outputs |
| Infrastructure | Docker Compose, AWS EC2, GitHub, Makefile | Reproducible build, cloud hosting, source control |

Deployment hardening reflected lessons from practice: the slim Python image needed `libgomp1` for XGBoost; the image was built on the x86_64 instance itself to avoid Apple-Silicon/amd64 mismatch; the security group exposes only SSH (developer IP) and port 8501; and the container restarts automatically on boot.

## 3.7 Business Impact Calculation

The retention opportunity is quantified with an explicit, adjustable accounting formula rather than a black-box estimate:

*retention opportunity = Σ (revenue at risk of targeted tier) × assumed save rate*

where per-customer revenue at risk is *p × monthly_fee × 12*. Save rates are stated assumptions, not measured effects; Section 4.4 reports the midpoint scenario and its sensitivity, and Section 4.6.3 flags the missing live experiment as a limitation.

## 3.8 Chapter Summary

The methodology combines a cleaned, warehouse-backed data layer (3.1); three engineered features with documented evidence of use (3.2); a four-model portfolio selected by cross-validation inside leak-free pipelines (3.3); a precisely stated probability chain — log-odds → sigmoid (verified exact) → isotonic calibration → business outputs (3.4); baselines, a leakage audit, and two validity diagnostics (3.5); a reproducible Dockerised architecture deployed on AWS (3.6); and a transparent business-impact accounting (3.7).

# 4. Results and Discussion

## 4.1 Descriptive Analytics of the Subscriber Base

The cleaned dataset (5,000 × 14, zero rows lost) yields the headline KPIs: 5,000 subscribers, MRR $68,417, churn rate 50.3%, average monthly fee $13.68. The delivered descriptive findings:

- **Plan tier matters.** Basic-plan customers churn most (≈62%) versus 45% Standard and 44% Premium.
- **Recency is decisive.** Churn rises monotonically with inactivity: Active (≤ 7 days) 13% → Lapsing (8–30 days) 31% → Dormant (> 30 days) 75%.
- **Engagement separates churners from stayers.** Churned customers watch far less than retained ones; the low-engagement tertile churns at ≈91%.
- **Financial impact.** $33,010 — 48.2% of MRR — was tied to customers who churned (visualised as a revenue waterfall).
- **Geography is immaterial.** Churn is nearly flat (48–52%) across all six regions and is explicitly not over-interpreted.
- **Where to focus.** The revenue-weighted priority view reconciles two perspectives: Basic has the highest churn *rate* while Premium carries the largest *revenue at risk*; the right target depends on whether the goal is rate reduction or revenue protection.

## 4.2 Model Performance

Table 4.1 — Held-out test performance (20%, stratified) versus naive baselines.

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.982 | 0.981 | 0.929 | 0.932 | 0.926 | 0.929 |
| Decision Tree | 0.978 | 0.982 | 0.953 | 0.977 | 0.930 | 0.954 |
| Random Forest | 0.998 | 0.998 | 0.984 | 0.988 | 0.980 | 0.984 |
| **XGBoost (selected by CV)** | **1.000** | **1.000** | **0.996** | **1.000** | **0.992** | **0.996** |
| *Baseline: majority class* | *0.503* | — | — | — | — | *0.503* |
| *Baseline: inactive ≥ 27 days* | *0.708* | — | — | — | — | *0.741* |

The calibrated XGBoost reaches test PR-AUC 0.9997 with Brier score 0.0054. The key churn drivers by feature importance are engagement and recency: `engagement_segment_High` (0.498) and `engagement_segment_Low` (0.243) rank first and second, `recency_bucket_Dormant` fourth; demographics contribute little. **Leakage audit:** retraining the selected model without `last_login_days` and `recency_bucket` moved PR-AUC from 0.9999 to 0.9969 — a delta of only 0.003 — so the near-perfect score is not a recency shortcut. Because the data is balanced 50/50, the usual "accuracy is misleading for rare churn" argument does *not* apply here; we lead with PR-AUC for business focus and transferability to real, imbalanced churn data.

## 4.3 Validity Diagnostics: Overfitting Check and Feature-Family Ablation

**Model status.** The deployed model is *unchanged*: it remains the CV-selected, isotonic-calibrated XGBoost. No retrained "fixed" model exists, deliberately — the open validity question is *temporal* (does low engagement precede churn as an early warning, or partly follow it as a post-churn artifact?), and a snapshot dataset with no timestamps cannot answer it; retraining on the same snapshot would change nothing. The honest corrective is disclosure plus the two diagnostics below, with temporal (longitudinal) data named as future work.

**Diagnostic 1 — the near-perfect scores are not overfitting.**

Table 4.2 — Train vs. cross-validation vs. held-out-test PR-AUC.

| Model | Train | CV (mean ± std) | Held-out test | Gap (train − test) |
|---|---|---|---|---|
| Logistic Regression | 0.986 | 0.984 (±0.002) | 0.982 | +0.004 |
| Decision Tree | 0.994 | 0.978 (±0.001) | 0.979 | +0.016 |
| Random Forest | 1.000 | 0.998 (±0.001) | 0.998 | +0.002 |
| XGBoost | 1.000 | 0.9995 (±0.0004) | 0.9999 | +0.0001 |

The generalization gap is near zero for every model and fold-to-fold variance is tiny; an overfit model shows a large train–test gap, not this. Decisively, even the ≈35-parameter Logistic Regression — far too small to memorize 4,000 training rows — scores 0.982, so the near-perfect results cannot be memorization: the classes really are that separable. Consistent with this, the deployed probability distribution is bimodal — 98.4% of the 5,000 calibrated probabilities are below 5% or above 95%, and only 4 customers fall in the Medium band (0.40–0.70) — and calibration (Brier 0.005) confirms these extreme probabilities are empirically *accurate* rather than over-confident.

**Diagnostic 2 — feature-family ablation localizes the signal.**

Table 4.3 — Feature-family ablation (XGBoost test PR-AUC).

| Feature set | Features removed / kept | Test PR-AUC |
|---|---|---|
| Full model | all 15 features kept | 1.000 |
| − recency family | drops `last_login_days`, `recency_bucket` | 0.997 |
| − engagement family | drops `avg_watch_time_per_day`, `engagement_segment`, `watch_hours`, `watch_per_profile` | 0.796 |
| Demographics/plan only | keeps age, gender, region, device, payment method, genre, plan, fee, profiles | 0.578 |
| *Random baseline* | — | *0.503* |

Essentially all predictive signal is behavioral: removing the engagement family collapses PR-AUC to 0.796, and demographics/plan alone (0.578) sit barely above random (0.503), while removing recency costs only 0.003. What a snapshot cannot tell us is the *direction* of the engagement signal — whether low viewing precedes cancellation (a genuine early warning) or partly follows the decision to leave (a post-churn artifact). The reported scores are therefore an **upper bound** on real-world early-warning performance. The proper fix, named explicitly as future work, is **point-in-time feature snapshots** (features as of month t, churn observed in month t+1) and **out-of-time validation** on longitudinal data.

## 4.4 Business Impact

Scoring all 5,000 customers with the calibrated model yields **2,507 high-risk customers**, total annual revenue at risk ≈ **$397,289**, and a mean predicted churn probability of 50.4% (consistent with the actual churn rate of 50.3%). The tier distribution is Low 2,489 / Medium 4 / High 2,507 — the near-empty Medium band is itself diagnostic of the saturated probabilities discussed in Section 4.3.

Applying the accounting formula of Section 3.7 to the high-risk tier, a 30% save rate corresponds to ≈ **$118,000/year** of retention opportunity on a $68k-MRR book. Table 4.4 shows sensitivity; the figures are decision-support references, not forecasts.

Table 4.4 — Retention-opportunity sensitivity to the save-rate assumption (high-risk tier).

| Save rate | 10% | 20% | 30% (midpoint) | 40% |
|---|---|---|---|---|
| Annual opportunity | ≈$39k | ≈$79k | ≈$118k | ≈$158k |

## 4.5 Page-by-Page Walkthrough

The deployed application is a Netflix-themed Streamlit dashboard of four pages, each filterable in real time (region, plan, gender, engagement level, age) and each mapped to a phase of Simon's framework.

### 4.5.1 Home Dashboard (Intelligence)

Executive overview: five KPIs (subscribers, MRR, churn rate, average fee, MRR tied to churners), key signals (Basic 62% / Dormant 75% / low-engagement 91%), churn overview charts, the revenue-weighted "where to focus" priority view, geography choropleth, revenue waterfall, and a predictive preview that appears once scored data exists.

### 4.5.2 Descriptive Analytics (Intelligence)

Five tabs — Churn Patterns, Engagement, Subscription Behavior, Demographics, Segments & Revenue — implementing the full descriptive layer: churned-versus-retained profiles, engagement-recency scatter, plan/payment/device composition donuts, gender pie and age-group bars, world-map choropleth, segmentation stacked bars, and revenue treemap.

### 4.5.3 Churn Risk (Design)

Predictive KPIs (high-risk count, predicted churn rate, revenue at risk), model comparison versus the naive baselines, feature-importance chart with engineered features highlighted, the churn-probability methodology, and calibration and leakage-audit metrics, with revenue at risk broken down by plan and tier.

### 4.5.4 Customer Simulation (Choice)

The full 5,000-row prediction table (probability, tier, revenue at risk, recommended action, illustrative key factor), plus a customer-level **what-if simulator**: build or load a customer profile and the trained calibrated model predicts churn probability live, with tier, revenue at risk, and action. Feature engineering for the single hypothetical row is guaranteed consistent with training by appending the simulated customer to the full population and re-deriving features on the combined frame. Table 4.5 shows real rows from the delivered table.

Table 4.5 — Example rows from the delivered prediction table (IDs truncated).

| Customer ID | Churn prob. | Risk tier | Revenue at risk | Key factor (illustrative) | Recommended action |
|---|---|---|---|---|---|
| `49a5df…` | 99.6% | High | $167.25/yr | Low engagement | Offer targeted discount / personalized content |
| `2f2b96…` | 69.3% | Medium | $74.77/yr | Stable usage | Send re-engagement campaign |
| `09746f…` | 9.2% | Low | $15.52/yr | Low recent activity | Maintain engagement / loyalty rewards |

## 4.6 Discussion

### 4.6.1 What the Numbers Actually Mean

The headline PR-AUC of ≈1.000 does not mean the system would predict real Netflix churn nearly perfectly; it means the synthetic classes are almost perfectly separable, as Section 4.3 demonstrates (bimodal probabilities, near-zero generalization gaps, an interpretable linear model at 0.982). The transferable results are the *methodology* — the leak-free protocol, baselines, calibration, and diagnostics — and the *structure* of the findings: engagement and recency dominate, demographics are immaterial, and roughly half of MRR sits with churners. The business figures ($397k at risk, $118k opportunity) are internally consistent accounting on the scored population, offered as decision-support references rather than forecasts.

### 4.6.2 Why the Deployed Model Remains XGBoost

XGBoost was selected by cross-validation and remains deployed, calibrated, and unchanged. We considered replacing it after the suspicious 1.000, but the evidence does not justify it: the overfitting check clears all models, the ablation shows the signal is behavioral rather than an artifact of any one feature, and the remaining doubt — post-churn contamination of watch-time features — applies equally to every model in the portfolio, including Random Forest (0.998), which we report as the more credible best *score* pending a feature-by-feature audit. Swapping models would change nothing about the underlying question; only longitudinal data can. Meanwhile the calibrated probabilities are demonstrably accurate on this data (Brier 0.005), which is what the downstream ranking and dollar-weighting require.

### 4.6.3 Threats to Validity

1. **Synthetic, balanced data.** Real churn is imbalanced and noisier; findings demonstrate methodology, not Netflix's actual churn structure.
2. **Near-perfect separability.** ≈1.000 PR-AUC would be unattainable in practice; scores are an upper bound (Section 4.3).
3. **No time dimension.** No tenure analysis, no time-series validation, single-snapshot warehouse; the early-warning-versus-artifact question is undecidable on this data.
4. **Flat geography.** Regional differences (48–52%) should not be over-interpreted.
5. **Heuristic key factors.** The per-customer "key factor" is a transparent illustrative rule, not a model explanation; per-customer SHAP is planned.
6. **Assumed save rates.** Campaign-impact figures are scenario arithmetic pending a live A/B experiment.

## 4.7 Chapter Summary

Descriptive analytics located churn in behavior (plan, engagement, recency) and quantified its cost (48.2% of MRR). Four models cleared both baselines by wide margins; the CV-selected XGBoost reached PR-AUC ≈1.000, which the validity diagnostics show is genuine separability of synthetic data rather than overfitting, while bounding what the score can claim. Scoring produced a prioritised, dollar-weighted customer list (2,507 high-risk; ≈$397k at risk; ≈$118k opportunity at a 30% save rate), delivered through a four-page deployed dashboard whose pages map onto Intelligence, Design, and Choice.

# 5. Conclusion and Recommendations

## 5.1 Conclusions

Mapped to the five objectives of Section 1.4:

1. **Customer characteristics and behavior — achieved.** Full demographic, behavioral, and subscription profiling with interactive filtering (five descriptive tabs).
2. **Churn patterns and engagement — achieved.** Plan tier, engagement, and recency identified as the dominant churn dimensions (Basic 62%; low-engagement 91%; Dormant 75%); geography shown to be immaterial.
3. **Revenue impact — achieved.** 48.2% of MRR ($33,010) tied to churners; revenue at risk per customer and segment; revenue-weighted priority view.
4. **Predictive churn analysis — achieved with disclosed caveats.** Four models under a leak-free, cross-validated protocol with baselines, isotonic calibration (Brier 0.005), a leakage audit (delta 0.003), an overfitting check (gaps ≤ +0.016), and a feature-family ablation localizing the signal in behavior; XGBoost selected by CV and retained deployed, with the scores framed as an upper bound on real-world performance.
5. **Interactive dashboard — achieved and deployed.** A four-page DSS mapped to Simon's Intelligence–Design–Choice framework, running live on AWS EC2 via Docker Compose, including a customer-level what-if simulator driven by the trained calibrated model.

## 5.2 Recommendations

### 5.2.1 For Retention Teams

Target by *dollars, not headcount*: within the high-risk tier, prioritise by revenue at risk, so a high-risk Premium subscriber outranks a high-risk Basic one. Treat recency as the operational tripwire — the Active → Lapsing transition (13% → 31% churn) is the cheapest intervention point — and treat the low-engagement segment (91% churn) as the standing re-engagement audience. Use the what-if simulator to sanity-check an offer before spending on it, and validate any assumed save rate with a controlled A/B experiment before scaling a campaign.

### 5.2.2 For Further DSS Development

Add per-customer SHAP explanations to replace the illustrative key-factor heuristic; attach an Elastic IP (or a managed host) for a stable demo address; extend the warehouse to a time-variant star schema when temporal fields exist; and wire an experimentation loop (campaign assignment, outcome capture) so save rates become measured quantities.

### 5.2.3 For Methodology in Similar Student Projects

Three lessons generalise. First, *treat perfect scores as bugs until proven otherwise*: the baselines, leakage audit, overfitting check, and ablation cost little and converted a suspicious 1.000 into a defensible, bounded claim. Second, *make the probability chain explicit* (raw score → sigmoid → calibration → business output, each step verified); it is the difference between a model demo and a decision system, and it survives instructor scrutiny. Third, *engineer for reproducibility from day one* (fixed seeds, one-command Docker rebuild, artifacts read by the app from a single source): most of our deployment problems — a missing `libgomp1` in the slim image, an arm64/x86_64 mismatch, unreliable hot-reload over a bind mount, a lost `.gitignore` sweeping build artifacts into git — were solved once and stayed solved because the environment was declarative.

## 5.3 Future Research Directions

The binding constraint of this study is the snapshot dataset. The priority direction is longitudinal: **point-in-time feature snapshots** (features as of month t, churn observed at t+1) with **out-of-time validation**, which would decide the early-warning-versus-artifact question that no amount of modeling on a snapshot can. Second, validation on real, imbalanced churn data (2–10% positive rate) to measure how far the methodology's rankings degrade under realistic noise. Third, uplift modeling — predicting *who is persuadable*, not merely who is at risk — as the natural successor to save-rate scenario arithmetic once experimental data exists. Fourth, a feature-by-feature audit for post-churn artifacts in the watch-time family, and cost-sensitive threshold optimisation tying tier boundaries to campaign economics rather than fixed cut-offs.

# REFERENCES

Ahmad, A. K., Jafar, A., & Aljoumaa, K. (2019). Customer churn prediction in telecom using machine learning in big data platform. *Journal of Big Data, 6*(1), 28.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM.

Davis, J., & Goadrich, M. (2006). The relationship between Precision-Recall and ROC curves. In *Proceedings of the 23rd International Conference on Machine Learning* (pp. 233–240). ACM.

Inmon, W. H. (2005). *Building the data warehouse* (4th ed.). Wiley.

Kimball, R., & Ross, M. (2013). *The data warehouse toolkit: The definitive guide to dimensional modeling* (3rd ed.). Wiley.

Lemmens, A., & Croux, C. (2006). Bagging and boosting classification trees to predict churn. *Journal of Marketing Research, 43*(2), 276–286.

Neslin, S. A., Gupta, S., Kamakura, W., Lu, J., & Mason, C. H. (2006). Defection detection: Measuring and understanding the predictive accuracy of customer churn models. *Journal of Marketing Research, 43*(2), 204–211.

Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In *Proceedings of the 22nd International Conference on Machine Learning* (pp. 625–632). ACM.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In A. J. Smola, P. Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), *Advances in large margin classifiers* (pp. 61–74). MIT Press.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432.

Simon, H. A. (1960). *The new science of management decision.* Harper & Brothers.

Vafeiadis, T., Diamantaras, K. I., Sarigiannidis, G., & Chatzisavvas, K. C. (2015). A comparison of machine learning techniques for customer churn prediction. *Simulation Modelling Practice and Theory, 55*, 1–9.

Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research, 218*(1), 211–229.

Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. In *Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 694–699). ACM.
