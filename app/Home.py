"""Netflix Retention DSS — Home: the descriptive command center (Simon: Intelligence).

The single descriptive view: global sidebar filters, KPIs, key signals, then
tabbed detail (overview & priority · engagement · behavior · demographics ·
revenue & segments) plus the predictive preview. The former Descriptive
Analytics page was folded in here (its code is preserved in app/hidden/).
Chart builders live in lib/charts.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # app/

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.charts import (
    DIM_OPTS,
    bar_count,
    churn_rate_bar,
    churn_waterfall,
    donut,
    grouped_status_bar,
    priority_bubble,
    priority_table,
    region_map,
    revenue_by_plan_bar,
    revenue_treemap,
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

df_all = load_clean()

# ------------------------------------------------------- global filters ----
st.sidebar.header("Filters")
regions = sorted(df_all["region"].unique())
plans = sorted(df_all["subscription_type"].unique())
genders = sorted(df_all["gender"].unique())
sel_region = st.sidebar.multiselect("Region", regions, default=regions)
sel_plan = st.sidebar.multiselect("Subscription plan", plans, default=plans)
sel_gender = st.sidebar.multiselect("Gender", genders, default=genders)
sel_seg = st.sidebar.multiselect("Engagement segment", SEGMENT_ORDER, default=SEGMENT_ORDER)
a_lo, a_hi = int(df_all["age"].min()), int(df_all["age"].max())
age_lo, age_hi = st.sidebar.slider("Age range", a_lo, a_hi, (a_lo, a_hi))

df = df_all[
    df_all["region"].isin(sel_region)
    & df_all["subscription_type"].isin(sel_plan)
    & df_all["gender"].isin(sel_gender)
    & df_all["engagement_segment"].isin(sel_seg)
    & df_all["age"].between(age_lo, age_hi)
]
if df.empty:
    st.warning("No customers match the current filters. Widen them in the sidebar.")
    st.stop()
df = add_status(df)
st.sidebar.caption(f"Showing **{len(df):,}** of {len(df_all):,} subscribers")

mrr = df["monthly_fee"].sum()
lost_mrr = df.loc[df["churned"] == 1, "monthly_fee"].sum()
overall_churn = df["churned"].mean() * 100


def _seg_rate(sub: pd.DataFrame) -> str:
    """Churn rate of a sub-population, '—' if the filters emptied it."""
    return f"{sub['churned'].mean() * 100:.0f}%" if len(sub) else "—"


# ------------------------------------------------------------------ KPIs ----
st.subheader("Key Performance Indicators")
c = st.columns(5)
c[0].metric("Total Subscribers", f"{len(df):,}")
c[1].metric("Total Revenue (MRR)", fmt_money(mrr))
c[2].metric("Overall Churn Rate", f"{overall_churn:.1f}%")
c[3].metric("Avg Monthly Fee", f"${df['monthly_fee'].mean():.2f}")
c[4].metric("MRR Tied to Churners", fmt_money(lost_mrr),
            delta=f"{lost_mrr / mrr * 100:.0f}% of MRR" if mrr else None,
            delta_color="inverse")

# -------------------------------------------------------------- signals ----
st.subheader("Key signals")
s = st.columns(3)
s[0].metric("Basic-plan churn", _seg_rate(df[df["subscription_type"] == "Basic"]),
            help="Highest-churning subscription plan.")
s[1].metric("Dormant-user churn", _seg_rate(df[df["recency_bucket"] == "Dormant"]),
            help="Customers inactive >30 days since last login.")
s[2].metric("Low-engagement churn", _seg_rate(df[df["engagement_segment"] == "Low"]),
            help="Bottom engagement tier (daily watch time).")
st.divider()

# ------------------------------------------------------------ tabbed detail ----
t_over, t_eng, t_behav, t_demo, t_rev = st.tabs([
    "📈 Overview", "🎬 Engagement", "💳 Behavior", "👤 Demographics",
    "💰 Revenue & Segments",
])

# ---- 📈 Overview: churn story + retention priority + waterfall --------------
with t_over:
    st.subheader("Churn overview")
    r1a, r1b = st.columns(2)
    with r1a:
        st.plotly_chart(donut(df, "Status", "Overall churn", cmap=CHURN_COLORS),
                        use_container_width=True)
    with r1b:
        st.plotly_chart(churn_rate_bar(df, "subscription_type", "Churn rate by plan"),
                        use_container_width=True)
    st.plotly_chart(grouped_status_bar(df), use_container_width=True)
    st.caption("Grouped bar: each feature's churned vs retained mean, divided by the "
               "overall average (1.0 = average). Churners log in far less recently "
               "and watch far less.")
    st.divider()

    st.subheader("Where to focus — retention priority")
    dim_label = st.radio("Break down by", list(DIM_OPTS), horizontal=True, key="priority_dim")
    tbl = priority_table(df, DIM_OPTS[dim_label])
    top = tbl.sort_values("arr_at_risk", ascending=False).iloc[0]
    st.info(f"🎯 **Focus here:** {dim_label.lower()} **{top['segment']}** — "
            f"{top['churn_rate']:.0f}% churn and {fmt_money(top['arr_at_risk'])}/yr at risk "
            f"across {int(top['subscribers']):,} subscribers.")
    f1, f2 = st.columns(2)
    with f1:
        st.plotly_chart(priority_bubble(tbl, dim_label, overall_churn), use_container_width=True)
    with f2:
        st.plotly_chart(risk_bar(tbl), use_container_width=True)
    st.caption("Bubble size = annual revenue lost to churn; dotted line = overall churn rate. "
               "Segments that are high (above the line) **and** large (right) are the top priorities.")
    st.divider()

    st.plotly_chart(churn_waterfall(df), use_container_width=True)

# ---- 🎬 Engagement -----------------------------------------------------------
with t_eng:
    st.subheader("Customer Engagement")
    fig = px.scatter(df, x="watch_hours", y="last_login_days", color="Status",
                     color_discrete_map=CHURN_COLORS, opacity=0.45,
                     title="Engagement vs recency (colour = churn status)",
                     labels={"watch_hours": "Watch hours",
                             "last_login_days": "Days since last login"},
                     category_orders={"Status": ["Retained", "Churned"]})
    fig.update_layout(margin=dict(t=46, b=10))
    st.plotly_chart(fig, use_container_width=True)
    e1, e2 = st.columns(2)
    with e1:
        st.plotly_chart(churn_rate_bar(df, "recency_bucket", "Churn rate by recency",
                                       order=RECENCY_ORDER), use_container_width=True)
    with e2:
        st.plotly_chart(churn_rate_bar(df, "engagement_segment",
                                       "Churn rate by engagement segment",
                                       order=SEGMENT_ORDER), use_container_width=True)
    st.caption("Scatter uses recency on the Y axis (no subscription-duration field exists). "
               "Retained users cluster in high-watch / low-recency; churners in the opposite corner.")

# ---- 💳 Behavior --------------------------------------------------------------
with t_behav:
    st.subheader("Subscription & Usage Behavior")
    st.caption("Each behavior dimension as its own donut (share of subscribers).")
    b1, b2 = st.columns(2)
    with b1:
        st.plotly_chart(donut(df, "subscription_type", "Plan type"), use_container_width=True)
        st.plotly_chart(donut(df, "device", "Device"), use_container_width=True)
    with b2:
        st.plotly_chart(donut(df, "payment_method", "Payment method"), use_container_width=True)
        st.plotly_chart(donut(df, "favorite_genre", "Favorite genre"), use_container_width=True)
    st.plotly_chart(donut(df, "number_of_profiles", "Number of profiles"),
                    use_container_width=True)

# ---- 👤 Demographics ----------------------------------------------------------
with t_demo:
    st.subheader("Customer Demographics")
    map_metric = st.radio("World map metric", ["Subscribers", "Churn %"],
                          horizontal=True, key="region_map_metric")
    st.plotly_chart(region_map(df, map_metric), use_container_width=True)
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(donut(df, "gender", "Gender", hole=0.0), use_container_width=True)  # pie
    with d2:
        df_band = df.assign(age_band=pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 200],
                            labels=["<25", "25-34", "35-44", "45-54", "55+"]))
        st.plotly_chart(bar_count(df_band, "age_band", "Age groups",
                                  order=["<25", "25-34", "35-44", "45-54", "55+"]),
                        use_container_width=True)
    st.plotly_chart(churn_rate_bar(df, "region", "Churn rate by region"),
                    use_container_width=True)
    st.caption("Gender → pie · Age groups → bar · Region → world-map heatmap.")

# ---- 💰 Revenue & Segments ----------------------------------------------------
with t_rev:
    st.subheader("Behavioral Segmentation & Revenue Impact")
    v1, v2 = st.columns(2)
    with v1:
        seg = df.groupby(["engagement_segment", "Status"]).size().reset_index(name="count")
        fig = px.bar(seg, x="engagement_segment", y="count", color="Status", barmode="stack",
                     color_discrete_map=CHURN_COLORS,
                     category_orders={"engagement_segment": SEGMENT_ORDER,
                                      "Status": ["Retained", "Churned"]},
                     title="Engagement segments (stacked by churn status)")
        fig.update_layout(xaxis_title="", margin=dict(t=46, b=10))
        st.plotly_chart(fig, use_container_width=True)  # stacked bar
    with v2:
        st.plotly_chart(revenue_treemap(df), use_container_width=True)  # treemap
    st.plotly_chart(revenue_by_plan_bar(df), use_container_width=True)
    st.caption("Stacked bar → customers per engagement segment (split by churn). "
               "Treemap → revenue share by plan & status.")

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
    st.caption("Predictive module scores the full population (sidebar filters don't apply "
               "here) — full detail on the Churn Risk page.")

# ---------------------------------------------------------------- nav ----
st.divider()
st.markdown("**Navigate** (sidebar): **Churn Risk** (Design) · "
            "**Customer Simulation** (Choice).")
