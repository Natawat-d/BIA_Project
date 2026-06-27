"""Netflix Retention DSS — Home: executive overview (Simon: Intelligence)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # app/

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.data_loader import has_clean, has_scored, load_clean, load_scored
from lib.viz import (
    CHURN_COLORS,
    NETFLIX_RED,
    RECENCY_ORDER,
    RISK_COLORS,
    RISK_ORDER,
    SEGMENT_ORDER,
    add_status,
    fmt_money,
)

st.set_page_config(page_title="Netflix Retention DSS", page_icon="🎬", layout="wide")

st.title("🎬 Netflix Customer Retention Intelligence — DSS")
st.caption("Methodology demonstration on a synthetic, ~50/50 balanced dataset · "
           "Simon framework: Intelligence → Design → Choice")

if not has_clean():
    st.warning("No processed data yet. Run `make etl` (or `python -m src.etl.load`).")
    st.stop()

df = add_status(load_clean())
mrr = df["monthly_fee"].sum()
lost_mrr = df.loc[df["churned"] == 1, "monthly_fee"].sum()


def churn_rate_bar(col, title, order=None):
    g = (df.groupby(col)["churned"].mean() * 100).reset_index()
    if order:
        g = g.set_index(col).reindex(order).reset_index()
    fig = px.bar(g, x=col, y="churned", title=title, color_discrete_sequence=[NETFLIX_RED])
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Churn %",
                      height=320, margin=dict(t=46, b=10))
    fig.update_traces(texttemplate="%{y:.0f}%", textposition="outside")
    return fig


# ------------------------------------------------------------------ KPIs ----
st.subheader("Key Performance Indicators")
c = st.columns(5)
c[0].metric("Total Subscribers", f"{len(df):,}")
c[1].metric("Total Revenue (MRR)", fmt_money(mrr))
c[2].metric("Overall Churn Rate", f"{df['churned'].mean() * 100:.1f}%")
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

# ------------------------------------------------------------ charts ----
r1a, r1b = st.columns(2)
with r1a:
    fig = px.pie(df, names="Status", hole=0.5, color="Status",
                 color_discrete_map=CHURN_COLORS, title="Overall churn")
    fig.update_layout(height=320, margin=dict(t=46, b=10))
    st.plotly_chart(fig, use_container_width=True)
with r1b:
    st.plotly_chart(churn_rate_bar("subscription_type", "Churn rate by plan"),
                    use_container_width=True)

r2a, r2b = st.columns(2)
with r2a:
    st.plotly_chart(churn_rate_bar("recency_bucket", "Churn rate by recency", order=RECENCY_ORDER),
                    use_container_width=True)
with r2b:
    st.plotly_chart(churn_rate_bar("engagement_segment", "Churn rate by engagement segment",
                                   order=SEGMENT_ORDER), use_container_width=True)

r3a, r3b = st.columns(2)
with r3a:
    rev = df.groupby("subscription_type")["monthly_fee"].sum().reset_index()
    fig = px.bar(rev, x="subscription_type", y="monthly_fee", title="Revenue (MRR) by plan",
                 color_discrete_sequence=[NETFLIX_RED])
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="MRR ($)",
                      height=320, margin=dict(t=46, b=10))
    st.plotly_chart(fig, use_container_width=True)
with r3b:
    retained = mrr - lost_mrr
    wf = go.Figure(go.Waterfall(
        measure=["absolute", "relative", "total"],
        x=["Total MRR", "Lost to churn", "Retained MRR"],
        text=[fmt_money(mrr), "−" + fmt_money(lost_mrr), fmt_money(retained)],
        textposition="outside", y=[mrr, -lost_mrr, retained],
        decreasing={"marker": {"color": NETFLIX_RED}},
        increasing={"marker": {"color": "#2E9E5B"}},
        totals={"marker": {"color": "#3A7CA5"}}))
    wf.update_layout(title="Financial impact of churn (MRR)", height=320,
                     yaxis_title="MRR ($)", margin=dict(t=46, b=10), showlegend=False)
    st.plotly_chart(wf, use_container_width=True)

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
