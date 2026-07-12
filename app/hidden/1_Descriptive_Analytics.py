"""Descriptive Analytics — Intelligence phase (maps to slide 7).

Chart types follow the agreed plan:
  KPI cards · donut (churn) · treemap (revenue by plan) · pie (gender) ·
  bar (age groups) · map (region) · donuts (subscription behavior) ·
  grouped bar (churned vs retained) · scatter (engagement) ·
  stacked bar (segmentation) · waterfall (financial impact of churn) ·
  retention-priority bubble (where to focus).
Purely descriptive — no model / prediction here. Chart builders live in lib/charts.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/

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
    revenue_treemap,
    risk_bar,
)
from lib.data_loader import has_clean, load_clean
from lib.viz import (
    CHURN_COLORS,
    RECENCY_ORDER,
    SEGMENT_ORDER,
    add_status,
    fmt_money,
    inject_theme_css,
)

st.set_page_config(page_title="Descriptive Analytics", page_icon="📊", layout="wide")
inject_theme_css()
st.title("📊 Descriptive Analytics — Understanding Our Customers")
st.caption("Intelligence phase · who our subscribers are, how they behave, and how churn differs across them")

if not has_clean():
    st.warning("No processed data yet. Run `make etl` (or `python -m src.etl.load`).")
    st.stop()

df_all = load_clean()

# ---------------------------------------------------------------- filters ----
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

# ------------------------------------------------------------------- KPIs ----
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Subscribers", f"{len(df):,}")
k2.metric("Total Revenue (MRR)", fmt_money(df["monthly_fee"].sum()))
k3.metric("Overall Churn Rate", f"{df['churned'].mean() * 100:.1f}%")
k4.metric("Avg Monthly Fee", f"${df['monthly_fee'].mean():.2f}")
k5.metric("Avg Watch Hours", f"{df['watch_hours'].mean():.1f}")
st.divider()

t_churn, t_eng, t_behav, t_demo, t_seg = st.tabs([
    "🔁 Churn Patterns", "🎬 Engagement", "💳 Subscription Behavior",
    "👤 Demographics", "💰 Segments & Revenue",
])

# ----------------------------------------------------- 6. Demographics --------
with t_demo:
    st.subheader("Customer Demographics")
    map_metric = st.radio("World map metric", ["Subscribers", "Churn %"],
                          horizontal=True, key="region_map_metric")
    st.plotly_chart(region_map(df, map_metric), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(donut(df, "gender", "Gender", hole=0.0), use_container_width=True)  # pie
    with c2:
        df_band = df.assign(age_band=pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 200],
                            labels=["<25", "25-34", "35-44", "45-54", "55+"]))
        st.plotly_chart(bar_count(df_band, "age_band", "Age groups",
                                  order=["<25", "25-34", "35-44", "45-54", "55+"]),
                        use_container_width=True)
    st.caption("Gender → pie · Age groups → bar · Region → world-map heatmap.")

# --------------------------------------------- 7. Subscription Behavior -------
with t_behav:
    st.subheader("Subscription & Usage Behavior")
    st.caption("Each behavior dimension as its own donut (share of subscribers).")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(donut(df, "subscription_type", "Plan type"), use_container_width=True)
        st.plotly_chart(donut(df, "device", "Device"), use_container_width=True)
    with c2:
        st.plotly_chart(donut(df, "payment_method", "Payment method"), use_container_width=True)
        st.plotly_chart(donut(df, "favorite_genre", "Favorite genre"), use_container_width=True)
    st.plotly_chart(donut(df, "number_of_profiles", "Number of profiles"), use_container_width=True)

# ---------------------------------------------------- 3+8. Churn Patterns -----
with t_churn:
    st.subheader("Churned vs Retained")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(donut(df, "Status", "Overall churn rate", cmap=CHURN_COLORS),
                        use_container_width=True)  # donut
    with c2:
        st.plotly_chart(grouped_status_bar(df), use_container_width=True)  # grouped bar
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(churn_rate_bar(df, "subscription_type", "Churn rate by plan"),
                        use_container_width=True)
    with c4:
        st.plotly_chart(churn_rate_bar(df, "region", "Churn rate by region"),
                        use_container_width=True)
    st.caption("Grouped bar: each feature's churned vs retained mean, divided by the overall "
               "average (1.0 = average). Churners log in far less recently and watch far less.")

# ------------------------------------------------------- 9. Engagement --------
with t_eng:
    st.subheader("Customer Engagement")
    fig = px.scatter(df, x="watch_hours", y="last_login_days", color="Status",
                     color_discrete_map=CHURN_COLORS, opacity=0.45,
                     title="Engagement vs recency (colour = churn status)",
                     labels={"watch_hours": "Watch hours", "last_login_days": "Days since last login"},
                     category_orders={"Status": ["Retained", "Churned"]})
    fig.update_layout(margin=dict(t=46, b=10))
    st.plotly_chart(fig, use_container_width=True)  # scatter
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(churn_rate_bar(df, "recency_bucket",
                                       "Churn rate by recency", order=RECENCY_ORDER),
                        use_container_width=True)
    with c2:
        st.plotly_chart(churn_rate_bar(df, "engagement_segment",
                                       "Churn rate by engagement segment", order=SEGMENT_ORDER),
                        use_container_width=True)
    st.caption("Scatter uses recency on the Y axis (no subscription-duration field exists). "
               "Retained users cluster in high-watch / low-recency; churners in the opposite corner.")

# ------------------------------------------- 10+5+11. Segments & Revenue ------
with t_seg:
    st.subheader("Retention Priority — where to focus")
    dim_label = st.radio("Break down by", list(DIM_OPTS), horizontal=True, key="priority_dim")
    tbl = priority_table(df, DIM_OPTS[dim_label])
    overall_churn = df["churned"].mean() * 100
    top = tbl.sort_values("arr_at_risk", ascending=False).iloc[0]
    st.info(f"🎯 **Focus here:** {dim_label.lower()} **{top['segment']}** — "
            f"{top['churn_rate']:.0f}% churn and {fmt_money(top['arr_at_risk'])}/yr at risk "
            f"across {int(top['subscribers']):,} subscribers.")
    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(priority_bubble(tbl, dim_label, overall_churn), use_container_width=True)
    with p2:
        st.plotly_chart(risk_bar(tbl), use_container_width=True)
    st.caption("Bubble size = annual revenue lost to churn; dotted line = overall churn rate. "
               "Segments that are high (above the line) **and** large (right) are the top priorities.")
    st.divider()

    st.subheader("Behavioral Segmentation & Revenue Impact")
    c1, c2 = st.columns(2)
    with c1:
        seg = df.groupby(["engagement_segment", "Status"]).size().reset_index(name="count")
        fig = px.bar(seg, x="engagement_segment", y="count", color="Status", barmode="stack",
                     color_discrete_map=CHURN_COLORS,
                     category_orders={"engagement_segment": SEGMENT_ORDER,
                                      "Status": ["Retained", "Churned"]},
                     title="Engagement segments (stacked by churn status)")
        fig.update_layout(xaxis_title="", margin=dict(t=46, b=10))
        st.plotly_chart(fig, use_container_width=True)  # stacked bar
    with c2:
        st.plotly_chart(revenue_treemap(df), use_container_width=True)  # treemap
    st.plotly_chart(churn_waterfall(df), use_container_width=True)  # waterfall
    st.caption("Stacked bar → customers per engagement segment (split by churn). "
               "Treemap → revenue share by plan & status. Waterfall → revenue lost to churn vs retained.")
