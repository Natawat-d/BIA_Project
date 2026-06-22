"""Churn Risk — Design phase (maps to slide 8)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.data_loader import has_scored, load_metrics, load_scored
from lib.viz import fmt_money

st.set_page_config(page_title="Churn Risk", page_icon="⚠️", layout="wide")
st.title("⚠️ Predictive Analytics — Churn Risk")
st.caption("Design phase · model-driven risk, revenue at risk, drivers, performance")

if not has_scored():
    st.warning("Run `make train` then `make score` first.")
    st.stop()

scored = load_scored()
metrics = load_metrics()

high = scored[scored["risk_tier"] == "High"]
rev_at_risk = scored["revenue_at_risk"].sum()
opportunity = high["revenue_at_risk"].sum() * 0.30  # assumed 30% save rate

c1, c2, c3, c4 = st.columns(4)
c1.metric("Predicted Churn Rate", f"{scored['churn_probability'].mean() * 100:.1f}%")
c2.metric("High-Risk Customers", f"{len(high):,}")
c3.metric("Revenue at Risk (yr)", fmt_money(rev_at_risk))
c4.metric("Retention Opportunity", fmt_money(opportunity),
          help="Σ high-risk revenue at risk × assumed 30% save rate")

st.divider()
left, right = st.columns(2)

with left:
    st.markdown("**Key Churn Drivers** — model feature importance")
    drivers = metrics.get("drivers", [])
    if drivers:
        dd = pd.DataFrame(drivers).sort_values("importance")
        st.plotly_chart(px.bar(dd, x="importance", y="feature", orientation="h"),
                        use_container_width=True)
    else:
        st.info("No driver data — run `make train`.")

with right:
    st.markdown("**Model Performance** — held-out test set")
    if metrics:
        st.write(f"Best model: **{metrics.get('best_model', '?')}**")
        rows = []
        for name, r in metrics.get("models", {}).items():
            t = r["test"]
            rows.append({"model": name, **{k: round(t[k], 3) for k in
                         ["pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"]}})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        bl = metrics.get("baselines", [])
        if bl:
            st.caption("Baselines — " +
                       "; ".join(f"{b['name']} (PR-AUC {b['pr_auc']:.2f})" for b in bl))
        la = metrics.get("leakage_audit")
        if la:
            delta = la["with_suspects"]["pr_auc"] - la["without_suspects"]["pr_auc"]
            st.caption(f"Leakage audit — PR-AUC changes by {delta:+.3f} when removing "
                       f"{', '.join(la['suspects'])}.")
    else:
        st.info("No metrics — run `make train`.")

st.divider()
st.markdown("**Revenue at Risk by Plan**")
rr = scored.groupby("subscription_type")["revenue_at_risk"].sum().reset_index()
st.plotly_chart(px.bar(rr, x="subscription_type", y="revenue_at_risk"),
                use_container_width=True)

st.caption("Note: drivers use model feature importance (no SHAP). The per-customer "
           "'key factor' shown on the next page is an illustrative heuristic.")
