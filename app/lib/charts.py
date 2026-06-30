"""Reusable Plotly chart builders shared by Home and the Descriptive page.

All figures inherit the Netflix dark template registered in lib.viz, so they
blend with the dark app theme. Builders take a DataFrame (already filtered /
with a 'Status' column where needed) and return a Plotly figure.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.viz import CHURN_COLORS, NETFLIX_RED, fmt_money

# ----------------------------------------------------------- simple builders --


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


# --------------------------------------------------------------- region map --
# Continent-level data → colour every country by its continent's value.
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
                      geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False,
                               showcoastlines=False, projection_type="natural earth"))
    return fig


# ------------------------------------------------------ churn vs retained ----


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


# ------------------------------------------------------------- revenue $$$ ----


def revenue_treemap(frame):
    rev = frame.groupby(["subscription_type", "Status"])["monthly_fee"].sum().reset_index()
    fig = px.treemap(rev, path=[px.Constant("All plans"), "subscription_type", "Status"],
                     values="monthly_fee", color="monthly_fee", color_continuous_scale="Reds",
                     title="Revenue (MRR) by plan")
    fig.update_layout(margin=dict(t=46, b=10))
    return fig


def revenue_by_plan_bar(frame):
    rev = frame.groupby("subscription_type")["monthly_fee"].sum().reset_index()
    fig = px.bar(rev, x="subscription_type", y="monthly_fee", title="Revenue (MRR) by plan",
                 color_discrete_sequence=[NETFLIX_RED])
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="MRR ($)",
                      margin=dict(t=46, b=10))
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


# -------------------------------------------------- retention priority view ---
# Break the book down by a manager-chosen dimension, scored on size, churn, and
# annual revenue currently lost to churn (a descriptive number — no model).
DIM_OPTS = {
    "Subscription plan": "subscription_type",
    "Region": "region",
    "Engagement segment": "engagement_segment",
    "Recency": "recency_bucket",
}


def priority_table(frame, dim):
    grp = frame.groupby(dim)
    out = pd.DataFrame({"subscribers": grp.size(),
                        "churn_rate": grp["churned"].mean() * 100})
    lost = frame.loc[frame["churned"] == 1].groupby(dim)["monthly_fee"].sum() * 12
    out["arr_at_risk"] = lost.reindex(out.index).fillna(0)
    out = out.reset_index().rename(columns={dim: "segment"})
    out["segment"] = out["segment"].astype(str)
    return out


def priority_bubble(tbl, dim_label, overall_churn):
    fig = px.scatter(tbl, x="subscribers", y="churn_rate", size="arr_at_risk",
                     color="segment", text="segment", size_max=60,
                     title=f"Retention priority by {dim_label.lower()}",
                     labels={"subscribers": "Subscribers", "churn_rate": "Churn %"})
    fig.add_hline(y=overall_churn, line_dash="dot",
                  annotation_text=f"overall churn {overall_churn:.0f}%")
    fig.update_traces(textposition="top center")
    fig.update_layout(showlegend=False, margin=dict(t=46, b=10),
                      yaxis_title="Churn %", xaxis_title="Subscribers")
    return fig


def risk_bar(tbl):
    t = tbl.sort_values("arr_at_risk")
    fig = px.bar(t, x="arr_at_risk", y="segment", orientation="h",
                 title="Annual revenue at risk", color_discrete_sequence=[NETFLIX_RED])
    fig.update_traces(texttemplate="$%{x:,.0f}", textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="Annual revenue at risk ($)",
                      yaxis_title="", margin=dict(t=46, b=10))
    return fig
