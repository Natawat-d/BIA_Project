"""Descriptive Analytics — Intelligence phase (maps to slide 7).

Chart types follow the agreed plan:
  KPI cards · donut (churn) · treemap (revenue by plan) · pie (gender) ·
  bar (age groups) · map (region) · donuts (subscription behavior) ·
  grouped bar (churned vs retained) · scatter (engagement) ·
  stacked bar (segmentation) · waterfall (financial impact of churn).
Purely descriptive — no model / prediction here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.data_loader import has_clean, load_clean
from lib.viz import (
    CHURN_COLORS,
    NETFLIX_RED,
    RECENCY_ORDER,
    SEGMENT_ORDER,
    add_status,
    fmt_money,
)

st.set_page_config(page_title="Descriptive Analytics", page_icon="📊", layout="wide")
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


# ---------------------------------------------------------------- helpers ----
def donut(frame, col, title, hole=0.5, cmap=None, order=None):
    vc = frame[col].astype(str).value_counts()
    if order:
        vc = vc.reindex([str(o) for o in order]).dropna()
    vc = vc.reset_index()
    vc.columns = [col, "count"]
    fig = px.pie(vc, names=col, values="count", hole=hole, title=title,
                 color=col if cmap else None, color_discrete_map=cmap)
    fig.update_layout(margin=dict(t=46, b=10))
    return fig


def bar_count(frame, col, title, color=NETFLIX_RED, order=None):
    vc = frame[col].astype(str).value_counts()
    if order:
        vc = vc.reindex([str(o) for o in order]).fillna(0)
    vc = vc.reset_index()
    vc.columns = [col, "count"]
    fig = px.bar(vc, x=col, y="count", title=title, color_discrete_sequence=[color])
    fig.update_layout(showlegend=False, xaxis_title="", margin=dict(t=46, b=10))
    return fig


def churn_rate_bar(frame, col, title, order=None):
    g = (frame.groupby(col)["churned"].mean() * 100).reset_index()
    if order:
        g = g.set_index(col).reindex(order).reset_index()
    fig = px.bar(g, x=col, y="churned", title=title, color_discrete_sequence=[NETFLIX_RED])
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Churn %",
                      margin=dict(t=46, b=10))
    fig.update_traces(texttemplate="%{y:.0f}%", textposition="outside")
    return fig


# Region heatmap (continent-level → colour every country by its continent's value).
_SOUTH_AMERICA = {"Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
                  "Paraguay", "Peru", "Uruguay", "Venezuela"}


@st.cache_data(show_spinner=False)
def _country_region_table():
    g = px.data.gapminder().query("year == 2007")[["country", "continent", "iso_alpha"]].copy()
    g["region"] = g.apply(
        lambda r: ("South America" if r["country"] in _SOUTH_AMERICA else "North America")
        if r["continent"] == "Americas" else r["continent"], axis=1)
    return g


def region_map(frame, metric):
    countries = _country_region_table()
    if metric == "Churn %":
        agg = (frame.groupby("region")["churned"].mean() * 100).round(1)
        label = "Churn %"
    else:
        agg = frame.groupby("region").size()
        label = "Subscribers"
    merged = countries.merge(agg.rename("value"), left_on="region", right_index=True, how="inner")
    fig = px.choropleth(merged, locations="iso_alpha", color="value", hover_name="region",
                        color_continuous_scale="Reds", labels={"value": label},
                        title=f"{label} by region")
    fig.update_layout(margin=dict(t=46, b=0, l=0, r=0), coloraxis_colorbar_title=label,
                      geo=dict(showframe=False, showcoastlines=False, projection_type="natural earth"))
    return fig


def grouped_status_bar(frame):
    feats = ["age", "monthly_fee", "watch_hours", "avg_watch_time_per_day",
             "last_login_days", "number_of_profiles"]
    means = frame.groupby("Status")[feats].mean()
    norm = means.div(frame[feats].mean()).T.reset_index()
    norm = norm.melt(id_vars="index", var_name="Status", value_name="ratio").rename(
        columns={"index": "feature"})
    fig = px.bar(norm, x="feature", y="ratio", color="Status", barmode="group",
                 color_discrete_map=CHURN_COLORS,
                 category_orders={"Status": ["Retained", "Churned"]},
                 title="Churned vs retained — average characteristics")
    fig.add_hline(y=1.0, line_dash="dot", annotation_text="overall average")
    fig.update_layout(xaxis_title="", yaxis_title="mean ÷ overall average", margin=dict(t=46, b=10))
    return fig


def revenue_treemap(frame):
    rev = frame.groupby(["subscription_type", "Status"])["monthly_fee"].sum().reset_index()
    fig = px.treemap(rev, path=[px.Constant("All plans"), "subscription_type", "Status"],
                     values="monthly_fee", color="monthly_fee", color_continuous_scale="Reds",
                     title="Revenue (MRR) by plan")
    fig.update_layout(margin=dict(t=46, b=10))
    return fig


def churn_waterfall(frame):
    total = frame["monthly_fee"].sum()
    lost = frame.loc[frame["churned"] == 1, "monthly_fee"].sum()
    retained = total - lost
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute", "relative", "total"],
        x=["Total MRR", "Lost to churn", "Retained MRR"],
        text=[fmt_money(total), "−" + fmt_money(lost), fmt_money(retained)],
        textposition="outside", y=[total, -lost, retained],
        decreasing={"marker": {"color": NETFLIX_RED}},
        increasing={"marker": {"color": "#2E9E5B"}},
        totals={"marker": {"color": "#3A7CA5"}},
    ))
    fig.update_layout(title="Financial impact of churn (monthly revenue)",
                      yaxis_title="MRR ($)", margin=dict(t=46, b=10), showlegend=False)
    return fig


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
