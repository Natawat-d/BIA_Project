# Netflix Customer Retention Intelligence — Decision Support System

## Final Project Report

**Course:** AT82.04 Business Intelligence and Analytics · Intersem 2026
**Institution:** Asian Institute of Technology — Dept. of Computer Science and Communication Technology
**Instructor:** Dr. Vatcharaporn Esichaikul

**Team:**
- Natawat Damrongsilp (st125841)
- Liza Shrestha (st126674)
- Subhana Chitrakar (st126138)

**Date:** July 2026

---

# 1. Introduction and Background

Subscription streaming revenue depends on recurring monthly fees, which makes customer churn a direct and compounding threat to revenue: every cancelled subscription removes its fee from Monthly Recurring Revenue (MRR) each month going forward. Retaining an existing customer is widely reported to be substantially cheaper than acquiring a new one, so the ability to identify *which* customers are likely to cancel — early enough to intervene — is a core business-intelligence capability for any subscription business.

This project builds a complete **Decision Support System (DSS)** for a Netflix-style streaming service. The system follows Simon's (1960) three-phase model of decision making: **Intelligence** (understand subscribers and how churn behaves), **Design** (build and validate machine-learning models that predict each customer's churn risk), and **Choice** (prioritise customers, recommend actions, and simulate what-if scenarios). The result is a fully working, containerised pipeline — ETL → Data Preparation → Model → Dashboard — deployed as a live, interactive web application.

**Primary end user:** the Customer Retention Manager. **Secondary users:** Marketing Manager, Customer Success Manager, and Subscription Business Manager.

# 2. Objectives

1. Analyze customer characteristics and subscription behavior.
2. Identify churn patterns and engagement trends (churned vs. retained; low/medium/high engagement).
3. Assess revenue impact and customer value.
4. Conduct predictive churn analysis with machine-learning models, identify key drivers, and evaluate rigorously.
5. Deploy an interactive retention-intelligence dashboard with risk scoring, segmentation, and customer-level what-if simulation.

# 3. Literature Review

**Churn prediction as supervised classification.** Predicting customer defection from historical behavioral attributes is a long-established application of supervised machine learning. Neslin et al. (2006) formalised the problem in their churn-modeling "tournament" and showed that model choice materially affects the economic value of a retention campaign. Lemmens and Croux (2006) demonstrated that ensemble tree methods (bagging and boosting) significantly outperform single classifiers for churn. Verbeke et al. (2012) extended this line with profit-driven churn modeling in telecommunications, arguing that churn models should be judged by business value rather than raw accuracy — a principle our system adopts through revenue-at-risk weighting. Comparative studies consistently find gradient-boosted trees among the strongest performers: Vafeiadis et al. (2015) benchmarked standard classifiers on telecom churn and found boosted ensembles superior, and Ahmad, Jafar and Aljoumaa (2019) reported XGBoost as the best model for telecom churn on a large real dataset. Our model portfolio — Logistic Regression, Decision Tree, Random Forest (Breiman, 2001), and XGBoost (Chen & Guestrin, 2016) — mirrors this literature: an interpretable linear baseline, a single tree, and two ensembles.

**Probability outputs and calibration.** Churn intervention requires a *ranked* list (whom to contact first) and *dollar weighting* (probability × revenue), so the model must output a probability, not a 0/1 label. Classifiers produce probabilities by mapping an unbounded raw score (log-odds) through the logistic (sigmoid) function; however, these raw probabilities are often poorly calibrated, especially for boosted trees, which tend to be over-confident (Niculescu-Mizil & Caruana, 2005). Two standard post-hoc corrections exist: Platt scaling, which fits a sigmoid to the scores (Platt, 1999), and isotonic regression, a non-parametric monotonic remapping (Zadrozny & Elkan, 2002). We use **isotonic calibration**, following Niculescu-Mizil and Caruana's (2005) finding that it is the more effective correction for tree ensembles when sufficient data is available.

**Evaluation metrics.** For classification where the positive class is the business focus, the precision-recall curve is more informative than the ROC curve, particularly under class imbalance (Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015). Although our dataset happens to be balanced, real churn populations are imbalanced (typically 2–10% monthly churn), so we lead with **PR-AUC** for methodological transferability, and report ROC-AUC, F1, precision, recall, accuracy, and the Brier score for completeness.

**DSS and data warehousing.** The system design follows Simon's (1960) Intelligence–Design–Choice framework for decision support. The BI storage layer follows the data-warehouse tradition: a subject-oriented, integrated, non-volatile analytical store (Inmon, 2005) organised as a fact table with analytical views, in the dimensional-modeling style of Kimball and Ross (2013).

# 4. Dataset

The working dataset is the *Netflix Customer Churn* dataset (Kaggle): **5,000 subscriber records × 14 columns**, with the binary target `churned` (1 = cancelled, 0 = retained).

| Property | Value |
|---|---|
| Rows / columns | 5,000 × 14 |
| Target | `churned` (0/1) |
| Class balance | ≈50/50 (2,515 churned / 2,485 retained; churn rate 50.3%) |
| Total MRR | $68,417 |
| Average monthly fee | $13.68 |
| Regions | 6 continents |
| Time dimension | **None** (no signup date / tenure column) |

**Raw columns.** Numeric: `age`, `watch_hours`, `last_login_days`, `monthly_fee`, `number_of_profiles`, `avg_watch_time_per_day`. Categorical: `gender`, `subscription_type` (Basic/Standard/Premium), `region`, `device`, `payment_method`, `favorite_genre`. Identifier: `customer_id` (excluded from modeling). Target: `churned`.

**Synthetic-data disclosure.** Netflix does not release real subscriber data; this dataset is synthetic and approximately balanced. We therefore frame the project as a **methodology demonstration on representative data**: the pipeline, validation protocol, and DSS design transfer to real data, but the near-perfect model scores in §8 would not (real churn is rarer, noisier, and harder to separate). This is disclosed rather than hidden, and revisited in §11 (Threats to Validity).

# 5. System Architecture

The system is a single reproducible pipeline — **ETL → Data Preparation → Model → Dashboard** — fully containerised with Docker Compose (three services: `pipeline`, `postgres`, `dashboard`) and deployed on an AWS EC2 instance (region ap-southeast-7, Bangkok). Every run is deterministic (fixed random seed 42).

| Layer | Technology | Role |
|---|---|---|
| Presentation | Streamlit + Plotly | Interactive multi-page dashboard |
| Application | Python 3.11 | Page scripts, data access, chart builders, caching |
| Analytics / ML | pandas, scikit-learn, XGBoost, joblib | ETL, feature engineering, training, calibration, scoring |
| Data & storage | PostgreSQL (data warehouse), CSV | Star-schema fact table + KPI views; processed pipeline outputs |
| Infrastructure | Docker Compose, AWS EC2, GitHub, Makefile | Reproducible build, cloud hosting, source control |

**Data warehouse.** The ETL Load step populates a PostgreSQL warehouse: a denormalised fact table `fact_subscriber` (one row per subscriber, primary key `customer_id`) plus three KPI views (`vw_kpi_summary`, `vw_revenue_by_plan`, `vw_churn_by_segment`). The store satisfies the classic warehouse characteristics — subject-oriented (subscriber churn), integrated (cleaned and standardised by ETL), non-volatile (read-only for analysis) (Inmon, 2005) — with two disclosed simplifications: it is a single snapshot (the dataset has no time dimension, so it is not time-variant), and a single wide fact table rather than a full star with separate dimension tables (Kimball & Ross, 2013). The dashboard reads the processed CSV outputs directly and falls back gracefully if the database is offline, so the demo always runs.

# 6. ETL Pipeline

**Extract** reads the raw CSV (5,000 × 14) with a fail-fast check that the file exists.

**Transform (data cleaning)** applies, in order: (1) standardise column names; (2) coerce the six numeric columns to numeric dtype, coercing invalid entries to missing; (3) drop duplicate `customer_id` rows; (4) impute missing values — median for numerics, "Unknown" for categoricals; (5) validate ranges (age 0–120; fee, watch hours, login recency, profiles ≥ 0) and drop violating rows; (6) normalise the target strictly to {0, 1}.

**Load** writes two destinations: `data/processed/clean.csv` (consumed by modeling and the dashboard) and the PostgreSQL data warehouse (fact table + KPI views), skipping the database gracefully when it is unavailable.

# 7. Data Preparation and Feature Engineering

## 7.1 How the engineered features are created

After cleaning, the preparation stage adds **three engineered features** (in `src/prep/features.py`), giving 15 model features (7 numeric + 8 categorical):

| Feature | Construction | Rationale |
|---|---|---|
| `watch_per_profile` | `watch_hours ÷ number_of_profiles` (safe against division by zero) | Normalises viewing volume by household size: distinguishes one heavy viewer from a large shared account. |
| `engagement_segment` | Population **tertiles** of `avg_watch_time_per_day` → Low / Medium / High | Converts a skewed continuous signal into a stable behavioral tier; matches how retention teams reason ("low-engagement users"). |
| `recency_bucket` | Fixed bands on `last_login_days`: Active (≤ 7), Lapsing (8–30), Dormant (> 30) | Encodes the classic recency dimension of RFM analysis as actionable, human-readable bands. |

All preprocessing (StandardScaler for numerics, one-hot encoding for categoricals — expanding 15 features to ≈35 model inputs) is wrapped inside a scikit-learn `Pipeline`, so it is fit on training folds only and cannot leak test information. The data is split 80:20 with **stratification** to preserve the churn distribution.

## 7.2 Are the engineered features actually used? — Evidence

Yes — on four independent grounds:

1. **They are model inputs**, one-hot encoded alongside the raw features (not display-only fields).
2. **They dominate the trained model's feature importance.** In the selected model, the top drivers are `engagement_segment_High` (importance 0.498) and `engagement_segment_Low` (0.243) — the two together account for **≈74% of total importance** — and `recency_bucket_Dormant` ranks fourth (0.024). The engineered features are not decorative; they are the model's primary signal carriers. (Feature-importance chart on the dashboard's Churn Risk page highlights engineered vs. raw features.)
3. **They structure the descriptive analytics.** Engagement segment and recency bucket are the axes of the dashboard's key findings — churn climbs from 13% (Active) to 31% (Lapsing) to 75% (Dormant), and the low-engagement tier churns at 91% — and both are sidebar filters applied across every page.
4. **They are stress-tested in the leakage audit** (§8.3): `recency_bucket` is one of the two recency suspects deliberately removed to test whether the model's performance is a recency artifact.

# 8. Predictive Modeling

## 8.1 Models and training protocol

Four classifiers are trained, each inside the leak-free pipeline: Logistic Regression (max_iter = 1000, class-balanced), Decision Tree (max depth 6), Random Forest (300 trees), and XGBoost (400 trees, depth 5, learning rate 0.05, subsample 0.9). Model selection uses **stratified k-fold cross-validation on the training set only** (metric: PR-AUC); the held-out test set is scored once for honest reporting. Two naive baselines anchor the results: majority class (PR-AUC 0.503) and a single-rule heuristic "inactive ≥ 27 days" (PR-AUC 0.708).

## 8.2 How the churn probability is computed (methodology)

This is the core of the predictive system, so we state it precisely:

1. **Raw score (log-odds).** Each classifier outputs an unbounded real-valued score. For Logistic Regression this is the linear combination *z = w·x + b*; for XGBoost it is the *margin*: a base value (initialised at the population churn rate) **plus the sum of all 400 boosted trees' leaf scores**, each tree having been fit to correct the residual errors of the trees before it (gradient boosting; Chen & Guestrin, 2016). We verified empirically on our data that the margin ranges roughly −15 to +17 and is exactly additive.
2. **Sigmoid link.** The logistic function *p = 1 / (1 + e^(−z))* maps the raw score to (0, 1). This is the standard link function for binary classification — not a normalisation of the score — and we verified that `sigmoid(raw score)` reproduces the library's `predict_proba` output exactly (maximum difference 0.0 across all 5,000 customers, for both Logistic Regression and XGBoost).
3. **Isotonic calibration.** Because boosted trees are characteristically over-confident (Niculescu-Mizil & Caruana, 2005), the selected model is wrapped in `CalibratedClassifierCV(method="isotonic", cv=5)` (Zadrozny & Elkan, 2002): a monotonic, non-parametric remapping learned on cross-validated predictions, so that a predicted 30% corresponds to an empirical churn rate of ≈30%. Calibration quality is measured by the Brier score (0.005 on the held-out test set).
4. **Business outputs.** The calibrated probability is then converted into decision-ready fields per customer: `risk_tier` (Low ≤ 0.40; High ≥ 0.70; Medium otherwise), `revenue_at_risk = p × monthly_fee × 12`, and a tier-mapped `recommended_action`.

**Why a probability rather than a 0/1 label?** A hard label answers only "will they churn?"; retention management needs "**who first, and how much is at stake?**" A calibrated probability supports (a) ranking customers for limited campaign budgets, (b) revenue weighting (p × fee), and (c) transparent, adjustable tier thresholds — none of which a binary label provides. This probability-first design follows established churn-modeling practice (Neslin et al., 2006; Verbeke et al., 2012), the model family follows the comparative evidence favouring boosted ensembles (Lemmens & Croux, 2006; Vafeiadis et al., 2015; Ahmad et al., 2019), and the calibration step follows the standard literature (Platt, 1999; Zadrozny & Elkan, 2002; Niculescu-Mizil & Caruana, 2005).

## 8.3 Results

**Held-out test set (20%, stratified):**

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.982 | 0.981 | 0.929 | 0.932 | 0.926 | 0.929 |
| Decision Tree | 0.978 | 0.982 | 0.953 | 0.977 | 0.930 | 0.954 |
| Random Forest | 0.998 | 0.998 | 0.984 | 0.988 | 0.980 | 0.984 |
| **XGBoost (selected by CV)** | **1.000** | **1.000** | **0.996** | **1.000** | **0.992** | **0.996** |
| *Baseline: majority class* | *0.503* | — | — | — | — | *0.503* |
| *Baseline: inactive ≥ 27 days* | *0.708* | — | — | — | — | *0.741* |

The calibrated XGBoost achieves test PR-AUC 0.9997 with **Brier score 0.0054**.

**Leakage audit.** Because recency is nearly a proxy for churn (a churned user stops logging in), we retrained the selected model **without** `last_login_days` and `recency_bucket`: PR-AUC moved from 0.9999 to 0.9969 — a delta of only **0.003**, showing the near-perfect score is not a recency shortcut.

**Honest checkpoint.** A PR-AUC of ≈1.000 is a red flag, not a triumph: scores this perfect almost always indicate that the data is too separable (or that some feature encodes the label). Real churn data would score materially lower. We therefore (a) disclose this prominently, (b) treat **Random Forest (0.998) as the more credible best model** pending a feature-by-feature audit for post-churn artifacts — the leading remaining suspect being `engagement_segment`/watch-time, since low viewing may be partly a *consequence* of churning rather than a cause — and (c) note that the calibration machinery already in the pipeline is the corrective for the observed probability saturation once any leak is removed.

**Balanced-data note.** Because this dataset is ≈50/50, the usual argument "accuracy is misleading because churn is a rare class" does **not** apply here; accuracy and F1 are legitimate. We lead with PR-AUC because the churn class and its ranking are the business focus, and because real churn data — where this methodology would be redeployed — is imbalanced.

## 8.4 Scoring output

Scoring every customer with the calibrated model produces `scored_customers.csv`: 2,507 high-risk customers, total annual revenue at risk ≈ $397,289, mean predicted churn probability 50.4% (internally consistent with the actual 50.3% churn rate). Each row carries the probability, tier, revenue at risk, a recommended action mapped from the tier, and an *illustrative* "key factor" tag produced by a transparent rule (recency → engagement → price, against population medians) — explicitly **not** a model explanation (no SHAP), a distinction stated in the interface.

# 9. The DSS Dashboard

The dashboard is a Netflix-themed Streamlit application; every page is filterable in real time.

| Page | Simon phase | Contents |
|---|---|---|
| **Home** | Intelligence | Executive overview: 5 KPIs (subscribers, MRR, churn rate, average fee, MRR tied to churners), key signals (Basic-plan 62% / Dormant 75% / low-engagement 91% churn), churn overview charts, "where to focus" retention-priority bubble, geography choropleth, revenue waterfall, predictive preview. |
| **Descriptive Analytics** | Intelligence | Five tabs — Churn Patterns, Engagement, Subscription Behavior, Demographics, Segments & Revenue — with sidebar filters (region, plan, gender, engagement, age) and a revenue-weighted retention-priority view. |
| **Churn Risk** | Design | Predictive KPIs, model comparison vs. baselines, feature-importance chart with engineered features highlighted, churn-probability methodology, calibration and leakage-audit metrics, revenue-at-risk by plan and tier. |
| **Customer Simulation** | Choice | 5,000-row customer prediction table (filterable), plus a **customer-level what-if simulator**: build or load a customer profile, and the trained calibrated model predicts their churn probability live, with risk tier, revenue at risk, and recommended action. |

**Key descriptive findings.** Basic-plan customers churn most (≈62% vs. 45% Standard / 44% Premium); churn rises monotonically with inactivity (Active 13% → Lapsing 31% → Dormant 75%); churned customers watch far less than retained ones; and **$33,010 — 48.2% of MRR — was tied to customers who churned**, which converts the churn rate into the financial stake. The revenue-weighted priority view reconciles two perspectives: Basic has the highest churn *rate*, while Premium carries the largest *revenue at risk* — the correct target depends on whether the goal is rate reduction or revenue protection.

**Deployment.** One command (`docker compose up`) reproduces the entire run; the stack is hosted live on AWS EC2 with the dashboard served over HTTP on port 8501.

# 10. Business Impact

The system quantifies the retention opportunity with an explicit, adjustable formula: *retention opportunity = Σ (revenue at risk of targeted tier) × assumed save rate*. At a 30% save rate on the high-risk tier this is ≈ **$118,000/year** on a $68k-MRR book. The save rate is an assumption, not a measured effect — validating it would require a live A/B test, which is stated in the interface and in §11. Sensitivity: at 10% / 20% / 40% save rates the opportunity is ≈ $39k / $79k / $158k respectively, scaling linearly.

# 11. Threats to Validity and Limitations

1. **Synthetic, balanced data.** The dataset is generated and ≈50/50 balanced; real churn is imbalanced and noisier. Findings demonstrate methodology, not Netflix's actual churn structure.
2. **Suspiciously perfect scores.** XGBoost's ≈1.000 PR-AUC indicates the classes are almost perfectly separable — expected in synthetic data, unattainable in practice. We report Random Forest as the credible best pending a full feature audit for post-churn artifacts (leading suspect: watch-time-derived features).
3. **No time dimension.** Without signup dates or event timestamps there is no tenure analysis, no time-series validation, and the data warehouse is a single snapshot (not time-variant).
4. **Geography is nearly flat** (churn 48–52% across all six regions), so regional patterns should not be over-interpreted.
5. **Key-factor tags are heuristic.** The per-customer "key factor" is a transparent illustrative rule, not a model explanation; per-customer SHAP explanations are a planned refinement.
6. **Save rates are assumptions.** Campaign-impact figures are scenario arithmetic; a live experiment would be required to measure true uplift.

# 12. Conclusions

Mapped to the five objectives:

1. **Customer characteristics & behavior — achieved.** Full demographic, behavioral, and subscription profiling with interactive filtering (Descriptive Analytics, five tabs).
2. **Churn patterns & engagement — achieved.** Plan tier, engagement segment, and recency identified as the dominant churn dimensions (Basic 62%; Low-engagement 91%; Dormant 75%); geography shown to be immaterial.
3. **Revenue impact — achieved.** 48.2% of MRR ($33,010) tied to churners; revenue-weighted priority view; churn waterfall; revenue at risk per customer and per segment.
4. **Predictive churn analysis — achieved with disclosed caveats.** Four models trained under a leak-free, cross-validated protocol with naive baselines, isotonic calibration (Brier 0.005), and a leakage audit (Δ 0.003); XGBoost selected by CV, Random Forest reported as the credible best pending a post-churn-artifact audit.
5. **Interactive dashboard — achieved and deployed.** Four-page Streamlit DSS mapped to Simon's Intelligence–Design–Choice, including a customer-level what-if simulator driven by the trained model, running live on AWS.

**Future work:** feature-by-feature post-churn-artifact audit; per-customer SHAP explanations; validation on real, imbalanced churn data; a time-variant warehouse once temporal fields exist; and an A/B-tested measurement of campaign save rates.

# References

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
