"""Customer Table & Retention Simulation — Choice phase (slides 9–10)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/

import streamlit as st

from lib.data_loader import has_scored, load_scored
from lib.viz import fmt_money

st.set_page_config(page_title="Retention Simulation", page_icon="🎯", layout="wide")
st.title("🎯 Customer Table & Retention Simulation")
st.caption("Choice phase · prioritise customers and test a campaign before spending")

if not has_scored():
    st.warning("Run `make train` then `make score` first.")
    st.stop()

scored = load_scored()

# ---- filters ---------------------------------------------------------------
st.sidebar.header("Filters")
plans = sorted(scored["subscription_type"].unique())
tiers = st.sidebar.multiselect("Risk tier", ["High", "Medium", "Low"],
                               default=["High", "Medium", "Low"])
sel_plans = st.sidebar.multiselect("Plan", plans, default=plans)
segs = st.sidebar.multiselect("Engagement", ["Low", "Medium", "High"],
                              default=["Low", "Medium", "High"])
max_rev = int(scored["revenue_at_risk"].max()) + 1
min_rev = st.sidebar.slider("Min revenue at risk ($/yr)", 0, max_rev, 0)

f = scored[
    scored["risk_tier"].isin(tiers)
    & scored["subscription_type"].isin(sel_plans)
    & scored["engagement_segment"].isin(segs)
    & (scored["revenue_at_risk"] >= min_rev)
]

# ---- customer prediction table --------------------------------------------
st.subheader(f"Customer Prediction Table — {len(f):,} customers")
show = f[["customer_id", "churn_probability", "risk_tier", "revenue_at_risk",
          "key_factor", "recommended_action"]].copy()
show["churn_probability"] = (show["churn_probability"] * 100).round(1)
show = show.sort_values("revenue_at_risk", ascending=False)
st.dataframe(
    show, use_container_width=True, hide_index=True,
    column_config={
        "churn_probability": st.column_config.NumberColumn("Churn %", format="%.1f"),
        "revenue_at_risk": st.column_config.NumberColumn("Revenue at risk", format="$%.0f"),
        "risk_tier": "Risk",
        "key_factor": "Key factor (heuristic)",
        "recommended_action": "Recommended action",
    },
)

# ---- what-if simulation ----------------------------------------------------
st.divider()
st.subheader("Retention Strategy Simulation (what-if)")
st.caption("Save rates are assumptions, not measured effects — a live A/B test "
           "would be required to validate them.")

a, b, c = st.columns(3)
target_tier = a.selectbox("Target group (risk tier)", ["High", "Medium", "Low"], index=0)
save_rate = b.slider("Assumed save rate", 0.0, 1.0, 0.30, 0.05)
discount = c.slider("Discount offered (%)", 0, 50, 20, 5) / 100.0

grp = scored[scored["risk_tier"] == target_tier]
expected_churners = grp["churn_probability"].sum()
baseline_rar = grp["revenue_at_risk"].sum()
customers_saved = expected_churners * save_rate
revenue_retained = baseline_rar * save_rate
campaign_cost = discount * grp["monthly_fee"].sum() * 12
net_benefit = revenue_retained - campaign_cost
overall_churners = scored["churn_probability"].sum()
new_churn_rate = (overall_churners - customers_saved) / len(scored)
base_churn_rate = scored["churn_probability"].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers Saved", f"{customers_saved:,.0f}")
m2.metric("Revenue Retained", fmt_money(revenue_retained))
m3.metric("Campaign Cost", fmt_money(campaign_cost))
m4.metric("Net Benefit", fmt_money(net_benefit))

st.metric(
    "New overall predicted churn rate",
    f"{new_churn_rate * 100:.1f}%",
    delta=f"{(new_churn_rate - base_churn_rate) * 100:.1f} pts",
    delta_color="inverse",
)
st.caption(f"Targeting **{len(grp):,}** {target_tier}-risk customers · "
           f"baseline revenue at risk in group = {fmt_money(baseline_rar)}.")
