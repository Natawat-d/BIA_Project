"""Netflix Retention DSS — Home: executive overview (Simon: Intelligence).

A single-screen command center: the KPIs, key signals, churn drivers, the
'where to focus' priority view, geography, and revenue impact — all the
important widgets in one place. Chart builders live in lib/charts.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # app/

import plotly.express as px
import streamlit as st

from lib.charts import (
    churn_rate_bar,
    churn_waterfall,
    donut,
    grouped_status_bar,
    priority_bubble,
    priority_table,
    region_map,
    revenue_by_plan_bar,
    risk_bar,
)
from lib.data_loader import has_clean, has_scored, load_clean, load_scored
from lib.viz import (
    CHURN_COLORS,
    RECENCY_ORDER,
    RISK_COLORS,
    RISK_ORDER,
    SEGMENT_ORDER,
    add_status,
    fmt_money,
    inject_theme_css,
)

st.set_page_config(page_title="Netflix Retention DSS", page_icon="🎬", layout="wide")
inject_theme_css()

st.title("🎬 Netflix Customer Retention Intelligence — DSS")
st.caption("Methodology demonstration on a synthetic, ~50/50 balanced dataset · "
           "Simon framework: Intelligence → Design → Choice")

if not has_clean():
    st.warning("No processed data yet. Run `make etl` (or `python -m src.etl.load`).")
    st.stop()

df = add_status(load_clean())
mrr = df["monthly_fee"].sum()
lost_mrr = df.loc[df["churned"] == 1, "monthly_fee"].sum()
overall_churn = df["churned"].mean() * 100

# ------------------------------------------------------------------ KPIs ----
st.subheader("Key Performance Indicators")
c = st.columns(5)
c[0].metric("Total Subscribers", f"{len(df):,}")
c[1].metric("Total Revenue (MRR)", fmt_money(mrr))
c[2].metric("Overall Churn Rate", f"{overall_churn:.1f}%")
c[3].metric("Avg Monthly Fee", f"${df['monthly_fee'].mean():.2f}")
c[4].metric("MRR Tied to Churners", fmt_money(lost_mrr),
            delta=f"{lost_mrr / mrr * 100:.0f}% of MRR", delta_color="inverse")

# -------------------------------------------------------------- signals ----
st.subheader("Key signals")
s = st.columns(3)
s[0].metric("Basic-plan churn", f"{df[df['subscription_type'] == 'Basic']['churned'].mean() * 100:.0f}%",
            help="Highest-churning subscription plan.")
s[1].metric("Dormant-user churn", f"{df[df['recency_bucket'] == 'Dormant']['churned'].mean() * 100:.0f}%",
            help="Customers inactive >30 days since last login.")
s[2].metric("Low-engagement churn", f"{df[df['engagement_segment'] == 'Low']['churned'].mean() * 100:.0f}%",
            help="Bottom engagement tier (daily watch time).")
st.divider()

# ----------------------------------------------------------- churn story ----
st.subheader("Churn overview")
r1a, r1b = st.columns(2)
with r1a:
    st.plotly_chart(donut(df, "Status", "Overall churn", cmap=CHURN_COLORS),
                    use_container_width=True)
with r1b:
    st.plotly_chart(churn_rate_bar(df, "subscription_type", "Churn rate by plan"),
                    use_container_width=True)

r2a, r2b = st.columns(2)
with r2a:
    st.plotly_chart(churn_rate_bar(df, "recency_bucket", "Churn rate by recency", order=RECENCY_ORDER),
                    use_container_width=True)
with r2b:
    st.plotly_chart(churn_rate_bar(df, "engagement_segment", "Churn rate by engagement segment",
                                   order=SEGMENT_ORDER), use_container_width=True)

st.plotly_chart(grouped_status_bar(df), use_container_width=True)
st.divider()

# ----------------------------------------------------------- where to focus ----
st.subheader("Where to focus — retention priority")
tbl = priority_table(df, "subscription_type")
top = tbl.sort_values("arr_at_risk", ascending=False).iloc[0]
st.info(f"🎯 **Top priority:** subscription plan **{top['segment']}** — "
        f"{top['churn_rate']:.0f}% churn and {fmt_money(top['arr_at_risk'])}/yr at risk "
        f"across {int(top['subscribers']):,} subscribers. "
        "Break this down by region / engagement / recency on the Descriptive page.")
f1, f2 = st.columns(2)
with f1:
    st.plotly_chart(priority_bubble(tbl, "Subscription plan", overall_churn), use_container_width=True)
with f2:
    st.plotly_chart(risk_bar(tbl), use_container_width=True)
st.divider()

# -------------------------------------------------------------- geography ----
st.subheader("Geography")
st.plotly_chart(region_map(df, "Churn %"), use_container_width=True)
st.divider()

# ---------------------------------------------------------- revenue impact ----
st.subheader("Revenue impact")
g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(revenue_by_plan_bar(df), use_container_width=True)
with g2:
    st.plotly_chart(churn_waterfall(df), use_container_width=True)

# ----------------------------------------------- predictive preview (opt) ----
if has_scored():
    st.divider()
    st.subheader("Predicted churn risk (preview)")
    scored = load_scored()
    p1, p2, p3 = st.columns(3)
    p1.metric("High-risk customers", f"{int((scored['risk_tier'] == 'High').sum()):,}")
    p2.metric("Revenue at risk (annual)", fmt_money(scored["revenue_at_risk"].sum()))
    p3.metric("Predicted churn rate", f"{scored['churn_probability'].mean() * 100:.1f}%")
    dist = (scored["risk_tier"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index())
    dist.columns = ["risk_tier", "count"]
    fig = px.bar(dist, x="risk_tier", y="count", color="risk_tier", title="Risk tier distribution",
                 color_discrete_map=RISK_COLORS, category_orders={"risk_tier": RISK_ORDER})
    fig.update_layout(showlegend=False, xaxis_title="", height=300, margin=dict(t=46, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Predictive module is a preview — full detail on the Churn Risk page (built later).")

# ---------------------------------------------------------------- nav ----
st.divider()
st.markdown("**Navigate** (sidebar): **📊 Descriptive Analytics** (Intelligence) · "
            "**Churn Risk** (Design) · **Retention Simulation** (Choice).")
