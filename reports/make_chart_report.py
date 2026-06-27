"""Generate Dashboard_Chart_Report.html — chart-type plan + interactive examples.

Run inside the container (has plotly + pandas):
  docker compose exec -T dashboard python /app/make_chart_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/app")
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.prep.features import add_engineered

ROOT = Path("/app")
df = add_engineered(pd.read_csv(ROOT / "data/processed/clean.csv"))
df["Status"] = df["churned"].map({0: "Retained", 1: "Churned"})

RED = "#E50914"
CHURN = {"Retained": "#3A7CA5", "Churned": "#E50914"}
SEG = ["Low", "Medium", "High"]
SA = {"Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
      "Paraguay", "Peru", "Uruguay", "Venezuela"}

KPI = dict(subs=f"{len(df):,}", rev=f"${df['monthly_fee'].sum():,.0f}",
           churn=f"{df['churned'].mean()*100:.1f}%", fee=f"${df['monthly_fee'].mean():.2f}")


def donut(col, title, hole=0.5, cmap=None):
    vc = df[col].astype(str).value_counts().reset_index()
    vc.columns = [col, "count"]
    f = px.pie(vc, names=col, values="count", hole=hole, title=title,
               color=col if cmap else None, color_discrete_map=cmap)
    return f


figs = {}
figs["churn"] = px.pie(df, names="Status", hole=0.5, color="Status",
                       color_discrete_map=CHURN, title="Overall churn rate")
rev = df.groupby(["subscription_type", "Status"])["monthly_fee"].sum().reset_index()
figs["revtree"] = px.treemap(rev, path=[px.Constant("All plans"), "subscription_type", "Status"],
                             values="monthly_fee", color="monthly_fee",
                             color_continuous_scale="Reds", title="Revenue (MRR) by plan")
figs["gender"] = donut("gender", "Gender", hole=0.0)
ab = pd.cut(df["age"], [0, 25, 35, 45, 55, 200],
            labels=["<25", "25-34", "35-44", "45-54", "55+"]).value_counts()
ab = ab.reindex(["<25", "25-34", "35-44", "45-54", "55+"]).reset_index()
ab.columns = ["age_band", "count"]
figs["age"] = px.bar(ab, x="age_band", y="count", title="Age groups", color_discrete_sequence=[RED])
# region map
g = px.data.gapminder().query("year == 2007")[["country", "continent", "iso_alpha"]].copy()
g["region"] = g.apply(lambda r: ("South America" if r["country"] in SA else "North America")
                      if r["continent"] == "Americas" else r["continent"], axis=1)
agg = df.groupby("region").size().rename("Subscribers")
m = g.merge(agg, left_on="region", right_index=True, how="inner")
figs["map"] = px.choropleth(m, locations="iso_alpha", color="Subscribers", hover_name="region",
                            color_continuous_scale="Reds", title="Subscribers by region")
figs["map"].update_layout(geo=dict(showframe=False, projection_type="natural earth"))
for col, key in [("subscription_type", "b_plan"), ("payment_method", "b_pay"),
                 ("device", "b_dev"), ("favorite_genre", "b_genre")]:
    figs[key] = donut(col, col.replace("_", " ").title())
feats = ["age", "monthly_fee", "watch_hours", "avg_watch_time_per_day",
         "last_login_days", "number_of_profiles"]
norm = df.groupby("Status")[feats].mean().div(df[feats].mean()).T.reset_index()
norm = norm.melt(id_vars="index", var_name="Status", value_name="ratio").rename(columns={"index": "feature"})
figs["grouped"] = px.bar(norm, x="feature", y="ratio", color="Status", barmode="group",
                         color_discrete_map=CHURN, title="Churned vs retained — avg characteristics (1.0 = average)")
figs["grouped"].add_hline(y=1.0, line_dash="dot")
figs["scatter"] = px.scatter(df, x="watch_hours", y="last_login_days", color="Status",
                             color_discrete_map=CHURN, opacity=0.45,
                             title="Engagement vs recency (colour = churn)",
                             labels={"last_login_days": "days since last login"})
seg = df.groupby(["engagement_segment", "Status"]).size().reset_index(name="count")
figs["seg"] = px.bar(seg, x="engagement_segment", y="count", color="Status", barmode="stack",
                     color_discrete_map=CHURN, category_orders={"engagement_segment": SEG},
                     title="Engagement segments (stacked by churn status)")
total = df["monthly_fee"].sum(); lost = df.loc[df.churned == 1, "monthly_fee"].sum()
figs["water"] = go.Figure(go.Waterfall(
    measure=["absolute", "relative", "total"], x=["Total MRR", "Lost to churn", "Retained MRR"],
    y=[total, -lost, total - lost], text=[f"${total:,.0f}", f"-${lost:,.0f}", f"${total-lost:,.0f}"],
    textposition="outside", decreasing={"marker": {"color": RED}},
    increasing={"marker": {"color": "#2E9E5B"}}, totals={"marker": {"color": "#3A7CA5"}}))
figs["water"].update_layout(title="Financial impact of churn (monthly revenue)")

for f in figs.values():
    f.update_layout(height=360, margin=dict(t=46, b=10, l=10, r=10))

_first = [True]
def div(key):
    inc = _first[0]
    _first[0] = False
    return figs[key].to_html(full_html=False, include_plotlyjs=inc)


CSS = """
*{box-sizing:border-box} body{font-family:"Segoe UI",Helvetica,Arial,sans-serif;color:#1a1a1a;
max-width:1000px;margin:0 auto;padding:30px 24px 60px;line-height:1.5}
h1{font-size:25px;margin:0 0 2px} .sub{color:#666;font-size:14px;margin-bottom:14px}
.bar{height:4px;background:#E50914;border-radius:2px;margin:6px 0 20px}
h2{color:#E50914;font-size:18px;margin:28px 0 6px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:6px 0 10px}
th,td{border:1px solid #e3e3e3;padding:7px 10px;text-align:left;vertical-align:top}
th{background:#fafafa} td.c{font-weight:600;color:#E50914;white-space:nowrap}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:6px 0}
.kpi{flex:1;min-width:180px;border:1px solid #e6e6e6;border-left:5px solid #E50914;border-radius:8px;padding:10px 14px}
.kpi .v{font-size:24px;font-weight:700}.kpi .l{color:#666;font-size:12px;text-transform:uppercase;letter-spacing:.4px}
.card{border:1px solid #e6e6e6;border-radius:10px;padding:14px 16px;margin:14px 0;break-inside:avoid;page-break-inside:avoid}
.card h3{display:inline;font-size:16px}.badge{display:inline-block;background:#E50914;color:#fff;font-size:11px;
text-transform:uppercase;letter-spacing:.5px;padding:2px 9px;border-radius:4px;margin-left:8px}
.card p{color:#444;font-size:13.5px;margin:8px 0 6px} .grid2{display:flex;gap:14px;flex-wrap:wrap}
.grid2>div{flex:1;min-width:320px} .note{font-size:13px;color:#666}
"""

rows = [
    ("1. Total Subscribers", "KPI Card", "Single headline number."),
    ("2. Total Revenue", "KPI Card", "Headline monthly recurring revenue."),
    ("3. Overall Churn Rate", "Donut", "Churned vs retained as parts of the whole."),
    ("4. Avg Subscription Duration", "KPI Card → <b>Avg Monthly Fee</b>",
     "No tenure/signup column in the data — substituted with Avg Monthly Fee."),
    ("5. Revenue by Plan", "Treemap", "Proportional revenue share by plan (and status)."),
    ("6. Customer Demographics", "Pie (gender) · Bar (age) · Map (region)",
     "Different demographic variables need different visuals."),
    ("7. Subscription Behavior", "Donuts (per dimension)",
     "No date column → donuts (share of subscribers) instead of a time-trend line."),
    ("8. Churned vs Retained", "Grouped Bar", "Compare average characteristics between the two groups."),
    ("9. Customer Engagement", "Scatter",
     "Engagement (watch hours) vs recency, coloured by churn (no duration field → recency)."),
    ("10. Behavioral Segmentation", "Stacked Bar", "Customers per engagement segment, split by churn."),
    ("11. Revenue & Value (impact of churn)", "Waterfall", "Total → lost to churn → retained revenue."),
]
table = "".join(f"<tr><td>{a}</td><td class='c'>{b}</td><td>{c}</td></tr>" for a, b, c in rows)


def card(title, badge, body_html, note):
    return (f'<div class="card"><h3>{title}</h3><span class="badge">{badge}</span>'
            f'<p>{note}</p>{body_html}</div>')


html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Dashboard Chart Report — Netflix Retention DSS</title><style>{CSS}</style></head><body>
<h1>Descriptive Dashboard — Chart Report</h1>
<div class="sub">Netflix Customer Retention Intelligence DSS · the chart type chosen for each
component (per the recommended plan) with a live, interactive example from the data (5,000 subscribers).</div>
<div class="bar"></div>

<h2>Chart plan (component → chart type)</h2>
<table><tr><th>Dashboard component</th><th>Chart used</th><th>Reason</th></tr>{table}</table>
<p class="note">Two items were adapted because the dataset is a single snapshot with no date or
tenure column: <b>#4</b> uses Avg Monthly Fee, and <b>#7</b> uses donuts instead of a time-trend line.</p>

<h2>1–4 · Headline numbers → KPI cards + donut</h2>
<div class="kpis">
 <div class="kpi"><div class="v">{KPI['subs']}</div><div class="l">Total Subscribers</div></div>
 <div class="kpi"><div class="v">{KPI['rev']}</div><div class="l">Total Revenue (MRR)</div></div>
 <div class="kpi"><div class="v">{KPI['churn']}</div><div class="l">Overall Churn Rate</div></div>
 <div class="kpi"><div class="v">{KPI['fee']}</div><div class="l">Avg Monthly Fee</div></div>
</div>
{card("3. Overall Churn Rate", "Donut", div("churn"), "Churned vs retained as parts of the whole.")}

<h2>5–6 · Revenue by plan & demographics</h2>
{card("5. Revenue by Plan", "Treemap", div("revtree"), "Proportional revenue share by plan, split by churn status.")}
{card("6. Demographics — Gender", "Pie", div("gender"), "Categorical split of subscribers by gender.")}
<div class="card"><h3>6. Demographics — Age & Region</h3><span class="badge">Bar + Map</span>
<p>Age groups as a bar; region as a world-map heatmap (every country coloured by its continent's value).</p>
<div class="grid2"><div>{div("age")}</div><div>{div("map")}</div></div></div>

<h2>7 · Subscription behavior → donuts</h2>
<div class="card"><h3>7. Subscription Behavior</h3><span class="badge">Donuts</span>
<p>Each behavior dimension as its own donut (share of subscribers).</p>
<div class="grid2"><div>{div("b_plan")}</div><div>{div("b_pay")}</div></div>
<div class="grid2"><div>{div("b_dev")}</div><div>{div("b_genre")}</div></div></div>

<h2>8–9 · Churn comparison & engagement</h2>
{card("8. Churned vs Retained", "Grouped Bar", div("grouped"),
      "Each feature's churned vs retained mean ÷ overall average (1.0 = average). Churners log in far less recently and watch far less.")}
{card("9. Customer Engagement", "Scatter", div("scatter"),
      "Watch hours vs days-since-login, coloured by churn. Retained cluster = high watch / low recency.")}

<h2>10–11 · Segmentation & financial impact</h2>
{card("10. Behavioral Segmentation", "Stacked Bar", div("seg"),
      "Customers per engagement segment, stacked by churn status.")}
{card("11. Revenue & Value — impact of churn", "Waterfall", div("water"),
      "Total monthly revenue → amount lost to churn → revenue retained.")}

<p class="note" style="margin-top:22px">Note: patterns look very clean because the dataset is synthetic
and roughly uniform across regions; real subscriber data would be noisier and more imbalanced.</p>
</body></html>"""

out = ROOT / "reports" / "Dashboard_Chart_Report.html"
out.write_text(html)
print("Wrote", out, "| bytes:", out.stat().st_size)
