"""Descriptive Analytics — Intelligence phase (maps to slide 7)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/

import plotly.express as px
import streamlit as st

from lib.data_loader import has_clean, load_clean
from lib.viz import SEGMENT_ORDER

st.set_page_config(page_title="Descriptive Analytics", page_icon="📊", layout="wide")
st.title("📊 Descriptive Analytics — Understanding Our Customers")
st.caption("Intelligence phase · who our subscribers are and how they churn")

if not has_clean():
    st.warning("Run `make etl` first.")
    st.stop()

df = load_clean()
tab1, tab2, tab3 = st.tabs(["Demographics & Plans", "Churned vs Retained", "Engagement"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(df, x="age", nbins=30, title="Age distribution"),
                        use_container_width=True)
        st.plotly_chart(px.pie(df, names="subscription_type", title="Subscription plan mix"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(df, x="region", color="gender", barmode="group",
                                     title="Region by gender"), use_container_width=True)
        st.plotly_chart(px.histogram(df, x="device", title="Device mix"),
                        use_container_width=True)
    st.plotly_chart(px.histogram(df, x="favorite_genre", title="Favorite genre"),
                    use_container_width=True)

with tab2:
    st.markdown("Churn rate by segment (mean of the 0/1 churn label).")
    c1, c2 = st.columns(2)
    with c1:
        by_plan = df.groupby("subscription_type")["churned"].mean().reset_index()
        st.plotly_chart(px.bar(by_plan, x="subscription_type", y="churned",
                               title="Churn rate by plan"), use_container_width=True)
    with c2:
        by_region = df.groupby("region")["churned"].mean().reset_index()
        st.plotly_chart(px.bar(by_region, x="region", y="churned",
                               title="Churn rate by region"), use_container_width=True)
    st.plotly_chart(
        px.histogram(df, x="watch_hours", color="churned", barmode="overlay", nbins=40,
                     title="Watch hours: churned vs retained"),
        use_container_width=True,
    )

with tab3:
    seg = (df.groupby("engagement_segment")["churned"].mean()
           .reindex(SEGMENT_ORDER).reset_index())
    st.plotly_chart(
        px.bar(seg, x="engagement_segment", y="churned",
               title="Churn rate by engagement segment",
               category_orders={"engagement_segment": SEGMENT_ORDER}),
        use_container_width=True,
    )
    st.plotly_chart(
        px.box(df, x="engagement_segment", y="avg_watch_time_per_day",
               title="Daily watch time by engagement segment",
               category_orders={"engagement_segment": SEGMENT_ORDER}),
        use_container_width=True,
    )
