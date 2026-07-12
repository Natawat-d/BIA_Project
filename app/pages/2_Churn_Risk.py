"""Churn Risk — Design phase (predictive analytics).

Predictive KPIs, model comparison vs baselines, feature importance (engineered
features highlighted), churn-probability methodology, calibration & leakage
audit, and revenue at risk. Reads reports/metrics.json + scored_customers.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.data_loader import has_scored, load_metrics, load_scored
from lib.viz import NETFLIX_RED, RISK_COLORS, RISK_ORDER, fmt_money, inject_theme_css

st.set_page_config(page_title="Churn Risk", page_icon="⚠️", layout="wide")
inject_theme_css()
st.title("⚠️ Predictive Analytics — Churn Risk")
st.caption("Design phase · model-driven risk, drivers, performance, and honesty checks")

if not has_scored():
    st.warning("Run `make train` then `make score` first.")
    st.stop()

scored = load_scored()
metrics = load_metrics()

# ------------------------------------------------------------------ KPIs ----
high = scored[scored["risk_tier"] == "High"]
rev_at_risk = scored["revenue_at_risk"].sum()
opportunity = high["revenue_at_risk"].sum() * 0.30  # assumed 30% save rate

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted Churn Rate", f"{scored['churn_probability'].mean() * 100:.1f}%")
c2.metric("High-Risk Customers", f"{len(high):,}")
c3.metric("Revenue at Risk (yr)", fmt_money(rev_at_risk))
c4.metric("Retention Opportunity", fmt_money(opportunity),
          help="Σ high-risk revenue at risk × assumed 30% save rate (assumption, not a measured effect)")
st.divider()

# ------------------------------------------- model comparison + drivers ----
ENGINEERED_PREFIXES = ("engagement_segment", "recency_bucket", "watch_per_profile")

left, right = st.columns(2)
with left:
    st.subheader("Model comparison — held-out test set")
    metric_label = st.selectbox("Metric", ["PR-AUC", "F1", "ROC-AUC", "Accuracy"], index=0)
    key = {"PR-AUC": "pr_auc", "F1": "f1", "ROC-AUC": "roc_auc", "Accuracy": "accuracy"}[metric_label]
    rows = [{"model": n, "value": r["test"][key], "kind": "ML model"}
            for n, r in metrics.get("models", {}).items()]
    for b in metrics.get("baselines", []):
        if key in b:
            rows.append({"model": b["name"], "value": b[key], "kind": "Naive baseline"})
    cmp_df = pd.DataFrame(rows).sort_values("value")
    fig = px.bar(cmp_df, x="value", y="model", orientation="h", color="kind",
                 color_discrete_map={"ML model": NETFLIX_RED, "Naive baseline": "#8A8A8A"},
                 title=f"Test {metric_label} — models vs naive baselines")
    fig.update_traces(texttemplate="%{x:.3f}", textposition="outside")
    fig.update_layout(height=380, margin=dict(t=46, b=10), xaxis_title=metric_label,
                      yaxis_title="", legend_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Best model by CV PR-AUC: **{metrics.get('best_model', '?')}** — "
               "selected on cross-validation, then isotonic-calibrated. Baselines show "
               "the ML models clearly earn their place.")

with right:
    st.subheader("Key churn drivers — feature importance")
    drivers = metrics.get("drivers", [])
    if drivers:
        dd = pd.DataFrame(drivers)
        dd["source"] = ["Engineered" if f.startswith(ENGINEERED_PREFIXES) else "Raw"
                        for f in dd["feature"]]
        dd = dd.sort_values("importance")
        fig = px.bar(dd, x="importance", y="feature", orientation="h", color="source",
                     color_discrete_map={"Engineered": NETFLIX_RED, "Raw": "#3A7CA5"},
                     title="Model feature importance (top 15)")
        fig.update_layout(height=380, margin=dict(t=46, b=10), yaxis_title="",
                          xaxis_title="importance", legend_title="Feature source")
        st.plotly_chart(fig, use_container_width=True)
        eng_share = dd.loc[dd["source"] == "Engineered", "importance"].sum()
        st.caption(f"**Engineered features (red) carry ≈{eng_share * 100:.0f}% of the total "
                   "importance** — engagement segment and recency bucket dominate. This is "
                   "direct evidence the feature engineering is used by the model, not decorative.")
    else:
        st.info("No driver data — run `make train`.")

# ----------------------------------------------- methodology (expander) ----
with st.expander("📐 How the churn probability is computed (methodology)"):
    st.markdown(
        """
1. **Features** — 15 inputs (12 raw + 3 engineered: `watch_per_profile`,
   `engagement_segment`, `recency_bucket`), one-hot encoded to ~35 columns inside a
   leak-free scikit-learn `Pipeline` (fit on training folds only).
2. **Raw score** — the classifier outputs a **log-odds score** (−∞…+∞):
   logistic regression = `w·x + b`; XGBoost = base + the sum of 400 boosted trees' leaf scores.
3. **Sigmoid** — `p = 1/(1+e^−score)` maps the raw score to a 0–1 probability
   (a link function, not normalisation).
4. **Isotonic calibration** — `CalibratedClassifierCV(method="isotonic", cv=5)` remaps
   the raw probability so a predicted 30% churns ~30% of the time (Brier ≈ 0.005).
5. **Business outputs** — `risk_tier` (Low ≤ 0.40 · High ≥ 0.70), and
   `revenue_at_risk = p × monthly_fee × 12`.

**Why a probability instead of a 0/1 label?** Retention needs a *ranking* (who first?)
and *dollar weighting* (p × fee) — a hard label supports neither. This mirrors standard
churn-prediction practice (Neslin et al. 2006; Verbeke et al. 2012; Ahmad et al. 2019 use
gradient boosting; calibration per Platt 1999 / Zadrozny & Elkan 2002). Full literature
review in the final report.
        """
    )

st.divider()

# --------------------------------------- distribution + honesty checks ----
h1, h2 = st.columns(2)
with h1:
    st.subheader("Churn probability distribution")
    fig = px.histogram(scored, x="churn_probability", nbins=40,
                       color_discrete_sequence=[NETFLIX_RED])
    fig.update_layout(height=320, margin=dict(t=20, b=10),
                      xaxis_title="Calibrated churn probability", yaxis_title="Customers")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Probabilities saturate near 0 and 1 — the synthetic data is highly "
               "separable, so the model is over-confident. Flagged in the honesty checkpoint.")
with h2:
    st.subheader("Rigor checks")
    cal = metrics.get("calibrated_test", {})
    la = metrics.get("leakage_audit", {})
    r1, r2 = st.columns(2)
    r1.metric("Calibrated PR-AUC", f"{cal.get('pr_auc', float('nan')):.3f}")
    r2.metric("Brier score", f"{cal.get('brier', float('nan')):.4f}",
              help="Mean squared error of the probabilities — lower is better-calibrated.")
    if la:
        delta = la["with_suspects"]["pr_auc"] - la["without_suspects"]["pr_auc"]
        r3, r4 = st.columns(2)
        r3.metric("Leakage audit Δ (PR-AUC)", f"{delta:+.3f}",
                  help=f"Score change after removing {', '.join(la['suspects'])}. "
                       "A tiny delta means recency is not the hidden shortcut.")
        r4.metric("Suspects removed", ", ".join(la["suspects"]))
    st.warning("**Honesty checkpoint:** XGBoost's ≈1.000 PR-AUC is a red flag, not a win — "
               "on real, imbalanced churn data scores would be materially lower. Random "
               "Forest (0.998) is the more credible best pending a full feature audit.")

# ------------------------------------------------- revenue at risk view ----
st.divider()
st.subheader("Revenue at risk — by plan and risk tier")
rr = scored.groupby(["subscription_type", "risk_tier"])["revenue_at_risk"].sum().reset_index()
fig = px.bar(rr, x="subscription_type", y="revenue_at_risk", color="risk_tier",
             barmode="stack", color_discrete_map=RISK_COLORS,
             category_orders={"risk_tier": RISK_ORDER})
fig.update_layout(height=340, margin=dict(t=20, b=10), xaxis_title="",
                  yaxis_title="Annual revenue at risk ($)", legend_title="Risk tier")
st.plotly_chart(fig, use_container_width=True)
st.caption("Drivers use model feature importance (no SHAP). The per-customer 'key factor' "
           "on the Customer Simulation page is an illustrative heuristic, not the model's reasoning.")
