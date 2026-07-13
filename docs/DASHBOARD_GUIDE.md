# Dashboard Explanation Report

**Netflix Customer Retention Intelligence DSS — what the dashboard shows, why each
widget exists, and how it helps the manager decide.**

Audience: the **Customer Retention Manager** (primary), plus the Marketing,
Customer Success, and Subscription Business Managers. Live demo:
`http://<host>:8501` (AWS EC2, Docker).

---

## 1. Purpose of the dashboard

Retention teams typically know *that* churn is high, but not **who to save
first, why, and how much is at stake**. This dashboard closes that gap. It is a
Decision Support System organised around Simon's three phases of decision
making, one page per phase:

| Page | Simon phase | The manager's question it answers |
|---|---|---|
| 🏠 Home | Intelligence | *What is the state of my subscriber base, and where is churn concentrated?* |
| ⚠️ Churn Risk | Design | *Can I trust the model's predictions, and what drives churn?* |
| 🎯 Customer Simulation | Choice | *Which individual customers do I act on, with what action — and what would change a customer's risk?* |

Every number on every page comes from one reproducible pipeline (ETL → features
→ model → scoring), so the descriptive views, the predictions, and the report
never disagree.

---

## 2. Home — the descriptive command center (Intelligence)

### 2.1 Global filters (sidebar)
Region, plan, gender, engagement segment, and age range — applied to **every**
KPI and chart on the page, with a "Showing X of Y subscribers" count.
**Why it matters:** the manager can ask any question scoped to their territory
("churn among European Basic-plan users under 35") without a data analyst.

### 2.2 KPI strip — the state of the book
| KPI | Meaning | Decision it informs |
|---|---|---|
| Total Subscribers | size of the (filtered) base | scale of any campaign |
| Total Revenue (MRR) | monthly recurring revenue | the money being defended |
| Overall Churn Rate | share of customers who cancelled | severity of the problem |
| Avg Monthly Fee | revenue per subscriber | value of a saved customer |
| **MRR Tied to Churners** | monthly revenue that belonged to customers who left (48% of MRR, ≈$33k) | converts the churn *rate* into a churn *cost* — the budget argument |

### 2.3 Key signals — the three levers
Three tiles name the highest-churn segments up front: **Basic plan ≈62%**,
**Dormant users (>30 days inactive) ≈75%**, **Low-engagement tier ≈91%**.
**Why:** a manager scanning for 10 seconds leaves knowing the three levers that
matter — plan tier, recency, engagement — before opening a single chart.

### 2.4 📈 Overview tab — churn story + where to focus
- **Churn donut + churn-by-plan bars** — the headline split and its strongest
  single dimension (Basic 62% vs Standard 45% / Premium 44%).
- **Churned-vs-retained profile bar** — for each attribute, the churned mean vs
  the retained mean (normalised to the population average). Shows *behaviour,
  not demographics,* separates leavers from stayers.
- **Retention priority view** (the "where do I focus?" widget): pick a
  dimension (plan / region / engagement / recency) →
  a **bubble matrix** (churn % × segment size, bubble = annual revenue lost)
  plus a **ranked revenue-at-risk bar**, and a 🎯 **"Focus here"** callout that
  names the single segment with the most recoverable revenue.
  **Why:** churn *rate* and churn *cost* disagree — Basic churns at the highest
  rate, but **Premium carries the most revenue at risk**. This widget resolves
  that tension explicitly instead of leaving it to intuition.
- **Churn waterfall** — Total MRR → lost to churn → retained MRR, in dollars.
  The financial-impact picture for an executive audience.

### 2.5 🎬 Engagement tab
Scatter of watch hours vs days-since-login (coloured by churn status) plus
churn-rate bars by recency band and engagement tier. **Why:** engagement is the
model's dominant driver; this tab shows the raw pattern behind it — retained
users cluster at high-watch/recent-login, churners at the opposite corner —
and gives the manager a defensible *intervention threshold* (churn jumps
13% → 31% → 75% across Active → Lapsing → Dormant).

### 2.6 💳 Behavior tab
Composition donuts: plan, device, payment method, favorite genre, profiles.
**Why:** campaign *targeting metadata* — which devices and payment methods the
at-risk population actually uses determines the channel (push vs e-mail vs
billing-page offer).

### 2.7 👤 Demographics tab
World-map choropleth (toggle: subscribers / churn %), gender pie, age bands,
churn by region. **Why:** mostly a *negative* finding worth knowing — churn is
nearly flat (48–52%) across regions, genders, and ages. The manager should
**not** spend budget on geographic or demographic targeting; the signal is
behavioural. Knowing where *not* to spend is a decision too.

### 2.8 💰 Revenue & Segments tab
Stacked engagement-segment bar (headcount split by churn status) and revenue
(MRR) by plan. **Why:** connects the behavioural segments to the money, and
shows which plan tier funds the business.

### 2.9 Predictive preview
High-risk customer count, annual revenue at risk, predicted churn rate, and
the risk-tier distribution — a teaser of the predictive layer with a pointer
to the Churn Risk page.

---

## 3. Churn Risk — trust and drivers (Design)

This page exists to answer the question every manager should ask before acting
on a model: **"why should I believe this?"**

- **Predictive KPIs** — predicted churn rate (50.4%, consistent with the actual
  50.3%), 2,507 high-risk customers, ≈$397k annual revenue at risk, and the
  **retention opportunity** (≈$118k/yr at an assumed 30% save rate — an
  explicit, adjustable assumption, not a promise).
- **Model comparison vs naive baselines** — four ML models against "always
  predict churn" (0.503) and "inactive ≥27 days" (0.708). **Why:** proves the
  ML earns its place over rules of thumb a manager could apply by hand.
- **Feature importance (engineered highlighted)** — the model's drivers, with
  the engineered features (engagement segment, recency bucket) in red carrying
  ≈74% of total importance. **Why:** tells the manager *what to change* —
  retention actions should target engagement and recency, because that is what
  the model says predicts leaving.
- **Methodology expander** — plain-language chain from features to probability
  (raw score → sigmoid → calibration). For the sceptical stakeholder.
- **Rigor checks** — calibration quality (Brier 0.005: a predicted 30% really
  churns ~30% of the time), the leakage audit (Δ 0.003), and an honesty
  warning: the near-perfect score reflects synthetic, highly separable data
  and should be read as an upper bound. **Why:** the dashboard tells the
  manager the model's *limits*, not just its strengths — that is what makes it
  a decision-support tool rather than a sales pitch.
- **Revenue at risk by plan × tier** — where the predicted losses sit, stacked
  by risk tier, for budget allocation.

---

## 4. Customer Simulation — act, customer by customer (Choice)

- **Customer Prediction Table** — all 5,000 customers with churn probability,
  risk tier, **revenue at risk** (probability × fee × 12), a heuristic "key
  factor" tag, and a **recommended action** mapped from the tier (High →
  targeted discount / personalised content; Medium → re-engagement campaign;
  Low → maintain). Filterable by tier/plan/engagement/minimum revenue, sorted
  by revenue at risk. **Why:** this is the working list — the manager's
  campaign export, ordered by money, not by raw probability.
- **Demo scenarios (10 pills)** — ten preset customer cases spanning 0%→100%
  risk, colour-coded by tier. One click loads a case. **Why:** in a meeting,
  the manager can *show* how risk behaves — the binge-watching loyalist scores
  0%, the ghost account 100%, and the interesting borderline cases in between.
- **What-if simulator** — every attribute of the selected customer is editable;
  the trained model re-predicts live on a gauge (tier bands + population
  average), with revenue at risk and the recommended action updating.
  **Why:** builds intuition for *sensitivity* — drop daily watch time and
  watch risk climb; the manager learns which levers move risk before spending
  on them. It also demystifies the model: predictions stop being a black box
  the moment you can push its buttons.

---

## 5. A retention manager's workflow on this dashboard

1. **Monday scan (Home, 2 min):** KPIs + key signals — is churn cost moving?
2. **Focus (Overview tab):** open the priority view, read the 🎯 callout —
   e.g., *"Premium: 44% churn, ~$160k/yr at risk across 1,693 subscribers."*
   Decide this week's target segment (rate-reduction vs revenue-protection).
3. **Trust check (Churn Risk, first visit / monthly):** baselines beaten,
   calibration good, drivers sensible → safe to act on the scores.
4. **Build the list (Customer Simulation):** filter High tier + target plan +
   minimum revenue at risk → the ranked outreach list with per-customer
   recommended actions.
5. **Sanity-check the intervention (what-if):** simulate the typical target
   customer; verify the levers the campaign pulls (engagement, recency)
   actually move the predicted risk.
6. **Make the budget case:** retention opportunity = revenue at risk × save
   rate ($39k / $79k / $118k / $158k at 10/20/30/40%) — an explicit formula
   the finance team can interrogate.

## 6. Value to the other managers

- **Marketing Manager:** behaviour/demographic composition for channel
  targeting; the negative geographic finding prevents wasted regional spend.
- **Customer Success Manager:** the Dormant/Lapsing recency bands define
  proactive-outreach triggers ("contact at day 8, escalate at day 30").
- **Subscription Business Manager:** plan-level economics — Basic's 62% churn
  rate vs Premium's revenue exposure — informs pricing and packaging reviews.

## 7. Honest limits (stated in the interface itself)

The data is synthetic and ~50/50 balanced; the model's near-perfect scores are
an upper bound; probabilities cluster at the extremes because the synthetic
classes are highly separable; the per-customer "key factor" is an illustrative
heuristic, not a model explanation; and save rates are assumptions pending a
live A/B test. These caveats appear on the pages where the relevant numbers
are shown — the dashboard is designed to keep its user appropriately sceptical.
