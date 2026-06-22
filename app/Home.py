"""Netflix Retention DSS — Home / KPI overview (Simon: Intelligence)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # app/

import plotly.express as px
import streamlit as st

from lib.data_loader import has_clean, has_scored, load_clean, load_scored
from lib.viz import RISK_COLORS, RISK_ORDER, fmt_money

st.set_page_config(page_title="Netflix Retention DSS", page_icon="🎬", layout="wide")

st.title("🎬 Netflix Customer Retention Intelligence — DSS")
st.caption(
    "Methodology demonstration on a synthetic, ~50/50 balanced dataset · "
    "Simon framework: Intelligence → Design → Choice"
)

if not has_clean():
    st.warning("No processed data yet. Run `make etl` (or `python -m src.etl.load`).")
    st.stop()

df = load_clean()

st.subheader("Key Performance Indicators")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Subscribers", f"{len(df):,}")
c2.metric("Total Revenue (MRR)", fmt_money(df["monthly_fee"].sum()))
c3.metric("Overall Churn Rate", f"{df['churned'].mean() * 100:.1f}%")
c4.metric("Avg Monthly Fee", f"${df['monthly_fee'].mean():.2f}")

st.divider()
left, right = st.columns(2)

with left:
    st.markdown("**Revenue by Plan**")
    rev = df.groupby("subscription_type")["monthly_fee"].sum().reset_index()
    fig = px.bar(rev, x="subscription_type", y="monthly_fee", color="subscription_type",
                 labels={"monthly_fee": "Revenue (MRR)", "subscription_type": "Plan"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("**Churn Risk Distribution**")
    if has_scored():
        dist = (
            load_scored()["risk_tier"].value_counts()
            .reindex(RISK_ORDER).fillna(0).reset_index()
        )
        dist.columns = ["risk_tier", "count"]
        fig = px.bar(dist, x="risk_tier", y="count", color="risk_tier",
                     color_discrete_map=RISK_COLORS,
                     category_orders={"risk_tier": RISK_ORDER})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `make train` then `make score` to populate predicted risk tiers.")

st.divider()
st.markdown(
    "**Navigate** (sidebar): **Descriptive Analytics** (Intelligence) · "
    "**Churn Risk** (Design) · **Retention Simulation** (Choice)."
)
