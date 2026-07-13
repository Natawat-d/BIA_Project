"""Customer Table & Churn Simulator — Choice phase.

Top: the scored customer prediction table (prioritisation).
Below: a demo panel of 10 PRESET customer cases spanning the risk spectrum —
all scored live through the trained calibrated model. Pick a case, see its
churn probability, then tweak behaviour/demographics and watch the risk move.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from joblib import load as joblib_load

from lib.data_loader import CLEAN, ROOT, has_clean, has_scored, load_scored
from lib.viz import NETFLIX_RED, RISK_COLORS, fmt_money, inject_theme_css

from src.config import ACTIONS, RISK_HIGH_MIN, RISK_LOW_MAX  # noqa: E402 (ROOT on path via data_loader)
from src.prep.features import add_engineered  # noqa: E402

MODEL_PATH = ROOT / "models" / "churn_model.pkl"

st.set_page_config(page_title="Customer Simulation", page_icon="🎯", layout="wide")
inject_theme_css()
st.title("🎯 Customer Table & Churn Simulator")
st.caption("Choice phase · prioritise customers, then explore 10 demo customer "
           "cases — each scored live by the trained model — and simulate what-if changes")

if not has_scored() or not MODEL_PATH.exists():
    st.warning("Run `make train` then `make score` first.")
    st.stop()

scored = load_scored()


@st.cache_resource(show_spinner=False)
def load_bundle():
    return joblib_load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_raw_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN)


bundle = load_bundle()
raw = load_raw_clean()

# ---- filters ---------------------------------------------------------------
st.sidebar.header("Table filters")
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

# ---- customer prediction table ---------------------------------------------
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

# ---- demo: 10 preset customer cases (pill selector + what-if) ---------------
st.divider()
st.subheader("Demo: 10 Customer Cases (what-if)")
st.caption("Ten preset customer profiles spanning the risk spectrum, all scored "
           "**live by the trained calibrated model** (nothing hardcoded) — click a "
           "scenario pill to auto-fill the controls, then tweak behaviour or "
           "demographics and watch the prediction respond. Engineered features "
           "(engagement segment, recency bucket) are derived exactly as in training.")

# Each persona is a full raw-feature row (validated against the dataset's
# category values and slider ranges). Behaviour values were tuned against the
# Platt-calibrated model so the panel spans Low -> Medium -> High risk with ten
# distinct scores (~0.7% to ~99.6%; verified 2026-07-13: 4 Low / 3 Medium / 3 High).
PERSONAS = [
    dict(name="🎬 Binge-watching Premium loyalist",
         blurb="4.5 h/day, logged in yesterday, four profiles — nothing to worry about.",
         age=29, gender="Female", subscription_type="Premium", monthly_fee=17.99,
         watch_hours=105.0, avg_watch_time_per_day=4.5, last_login_days=1,
         number_of_profiles=4, region="Asia", device="TV",
         payment_method="Credit Card", favorite_genre="Sci-Fi"),
    dict(name="🎓 Weekend student viewer",
         blurb="Light Basic plan on mobile, watches at weekends but keeps coming back.",
         age=22, gender="Female", subscription_type="Basic", monthly_fee=8.99,
         watch_hours=18.0, avg_watch_time_per_day=0.6, last_login_days=6,
         number_of_profiles=1, region="Europe", device="Mobile",
         payment_method="Debit Card", favorite_genre="Romance"),
    dict(name="🏖️ Devoted fan on a long break",
         blurb="35 days away, but a strong viewing history keeps her risk low.",
         age=63, gender="Female", subscription_type="Standard", monthly_fee=13.99,
         watch_hours=15.3, avg_watch_time_per_day=0.51, last_login_days=35,
         number_of_profiles=2, region="Oceania", device="Desktop",
         payment_method="Credit Card", favorite_genre="Documentary"),
    dict(name="👨‍👩‍👧 Family plan, cooling off",
         blurb="Three profiles on the TV but almost nothing watched lately — one to watch.",
         age=41, gender="Male", subscription_type="Standard", monthly_fee=13.99,
         watch_hours=1.0, avg_watch_time_per_day=0.12, last_login_days=4,
         number_of_profiles=3, region="North America", device="TV",
         payment_method="PayPal", favorite_genre="Comedy"),
    dict(name="🌗 On-the-fence newcomer",
         blurb="Logged in yesterday but has barely watched anything — could go either way.",
         age=26, gender="Other", subscription_type="Standard", monthly_fee=13.99,
         watch_hours=1.0, avg_watch_time_per_day=0.08, last_login_days=1,
         number_of_profiles=2, region="South America", device="Mobile",
         payment_method="Credit Card", favorite_genre="Horror"),
    dict(name="⚠️ Fading Premium subscriber",
         blurb="Top-tier fee, minimal viewing — high value with a genuinely uncertain score.",
         age=48, gender="Male", subscription_type="Premium", monthly_fee=17.99,
         watch_hours=1.0, avg_watch_time_per_day=0.12, last_login_days=5,
         number_of_profiles=2, region="North America", device="Laptop",
         payment_method="PayPal", favorite_genre="Drama"),
    dict(name="💤 Lapsing casual viewer",
         blurb="Low engagement, four days since login — right at the High-risk doorstep.",
         age=35, gender="Male", subscription_type="Standard", monthly_fee=13.99,
         watch_hours=1.0, avg_watch_time_per_day=0.12, last_login_days=4,
         number_of_profiles=2, region="Europe", device="Laptop",
         payment_method="PayPal", favorite_genre="Action"),
    dict(name="📉 Once-regular, tapering fast",
         blurb="Still logs in, but viewing has collapsed — the model is fairly sure he's leaving.",
         age=45, gender="Male", subscription_type="Standard", monthly_fee=13.99,
         watch_hours=3.0, avg_watch_time_per_day=0.14, last_login_days=4,
         number_of_profiles=2, region="Africa", device="TV",
         payment_method="Credit Card", favorite_genre="Action"),
    dict(name="💳 Crypto-paying minimalist",
         blurb="Minimal viewing plus a churn-prone payment method — near-certain churn.",
         age=52, gender="Male", subscription_type="Standard", monthly_fee=13.99,
         watch_hours=1.0, avg_watch_time_per_day=0.12, last_login_days=5,
         number_of_profiles=2, region="Asia", device="Desktop",
         payment_method="Crypto", favorite_genre="Sci-Fi"),
    dict(name="👻 Ghost account",
         blurb="Gift-card signup, zero watching, 60 days dormant — textbook churn.",
         age=33, gender="Other", subscription_type="Basic", monthly_fee=8.99,
         watch_hours=0.2, avg_watch_time_per_day=0.0, last_login_days=60,
         number_of_profiles=1, region="South America", device="Mobile",
         payment_method="Gift Card", favorite_genre="Horror"),
]

RAW_COLS = ["age", "gender", "subscription_type", "watch_hours", "last_login_days",
            "region", "device", "monthly_fee", "payment_method",
            "number_of_profiles", "avg_watch_time_per_day", "favorite_genre"]
TIER_DOT = {"Low": "🟢 Low", "Medium": "🟠 Medium", "High": "🔴 High"}


def tier_of(p: float) -> str:
    return "High" if p >= RISK_HIGH_MIN else ("Low" if p <= RISK_LOW_MAX else "Medium")


@st.cache_data(show_spinner=False)
def score_personas(personas_df: pd.DataFrame) -> pd.DataFrame:
    """Score the demo panel through the real model (engineered vs the population)."""
    rows = personas_df[RAW_COLS].copy()
    rows.insert(0, "customer_id", [f"DEMO-{i + 1:02d}" for i in range(len(rows))])
    rows["churned"] = 0
    pop = load_raw_clean()
    b = load_bundle()
    eng = add_engineered(pd.concat([pop, rows], ignore_index=True)).tail(len(rows))
    probs = b["model"].predict_proba(eng[b["num"] + b["cat"]])[:, 1]
    out = personas_df.copy()
    out["churn_probability"] = probs
    out["risk_tier"] = [tier_of(p) for p in probs]
    out["revenue_at_risk"] = out["churn_probability"] * out["monthly_fee"] * 12
    return out


panel = score_personas(pd.DataFrame(PERSONAS))

names = list(panel["name"])
if st.session_state.get("demo_persona") not in names:
    st.session_state["demo_persona"] = names[0]

# ---- demo-scenario pills: numbered 1-10 (panel order = lowest -> highest risk),
# colour-coded by risk tier; clicking one selects that persona ----------------
st.caption("**DEMO SCENARIOS** — click to auto-fill and test")
pill_cols = st.columns(10)
for i, (col, (_, prow)) in enumerate(zip(pill_cols, panel.iterrows()), start=1):
    with col:
        if st.button(str(i), key=f"demo_pill_{i}", use_container_width=True,
                     help=(f"{prow['name']} — {prow['churn_probability'] * 100:.1f}% churn "
                           f"({prow['risk_tier']} risk)")):
            st.session_state["demo_persona"] = prow["name"]

# style the pills AFTER the clicks are processed so the highlight is current
_css = ["<style>"]
for i, (_, prow) in enumerate(panel.iterrows(), start=1):
    color = RISK_COLORS[prow["risk_tier"]]
    base = f".st-key-demo_pill_{i} button"
    if prow["name"] == st.session_state["demo_persona"]:  # selected -> solid fill
        _css.append(f"{base}{{background:{color};border:2px solid {color};color:#141414;"
                    f"font-weight:800;border-radius:999px;min-height:2.1rem;padding:0;}}")
        _css.append(f"{base}:hover,{base}:focus:not(:active)"
                    f"{{background:{color};border-color:{color};color:#141414;}}")
    else:  # outlined chip in the tier colour
        _css.append(f"{base}{{background:transparent;border:1px solid {color};color:{color};"
                    f"font-weight:700;border-radius:999px;min-height:2.1rem;padding:0;}}")
        _css.append(f"{base}:hover,{base}:focus:not(:active)"
                    f"{{border:2px solid {color};color:{color};background:transparent;}}")
_css.append("</style>")
st.markdown("".join(_css), unsafe_allow_html=True)

choice = st.session_state["demo_persona"]
sel = panel.loc[panel["name"] == choice].iloc[0]
base_row = sel.to_dict()
st.markdown(f"**{choice}** — *{base_row['blurb']}* · panel score: "
            f"**{base_row['churn_probability'] * 100:.1f}%** "
            f"({TIER_DOT[base_row['risk_tier']]})")
k = choice  # widget keys include the persona so changing it resets the widgets

c1, c2, c3 = st.columns(3)
with c1:
    age = st.slider("Age", int(raw["age"].min()), int(raw["age"].max()),
                    int(base_row["age"]), key=f"age{k}")
    gender = st.selectbox("Gender", sorted(raw["gender"].unique()),
                          index=sorted(raw["gender"].unique()).index(base_row["gender"]),
                          key=f"g{k}")
    plan = st.selectbox("Subscription plan", sorted(raw["subscription_type"].unique()),
                        index=sorted(raw["subscription_type"].unique()).index(base_row["subscription_type"]),
                        key=f"p{k}")
    fee = st.slider("Monthly fee ($)", float(raw["monthly_fee"].min()),
                    float(raw["monthly_fee"].max()), float(base_row["monthly_fee"]),
                    0.5, key=f"f{k}")
with c2:
    watch = st.slider("Total watch hours", float(raw["watch_hours"].min()),
                      float(raw["watch_hours"].max()), float(base_row["watch_hours"]),
                      0.1, key=f"w{k}")
    daily = st.slider("Avg watch time / day (h)", float(raw["avg_watch_time_per_day"].min()),
                      float(raw["avg_watch_time_per_day"].max()),
                      float(base_row["avg_watch_time_per_day"]), 0.01, key=f"d{k}")
    login = st.slider("Days since last login", int(raw["last_login_days"].min()),
                      int(raw["last_login_days"].max()), int(base_row["last_login_days"]),
                      key=f"l{k}")
    profiles = st.slider("Number of profiles", int(raw["number_of_profiles"].min()),
                         int(raw["number_of_profiles"].max()),
                         int(base_row["number_of_profiles"]), key=f"n{k}")
with c3:
    region = st.selectbox("Region", sorted(raw["region"].unique()),
                          index=sorted(raw["region"].unique()).index(base_row["region"]),
                          key=f"r{k}")
    device = st.selectbox("Device", sorted(raw["device"].unique()),
                          index=sorted(raw["device"].unique()).index(base_row["device"]),
                          key=f"dev{k}")
    pay = st.selectbox("Payment method", sorted(raw["payment_method"].unique()),
                       index=sorted(raw["payment_method"].unique()).index(base_row["payment_method"]),
                       key=f"pay{k}")
    genre = st.selectbox("Favorite genre", sorted(raw["favorite_genre"].unique()),
                         index=sorted(raw["favorite_genre"].unique()).index(base_row["favorite_genre"]),
                         key=f"gen{k}")

# ---- predict: engineer features against the real population, then score ----
sim = pd.DataFrame([{
    "customer_id": "SIMULATED", "age": age, "gender": gender,
    "subscription_type": plan, "watch_hours": watch, "last_login_days": login,
    "region": region, "device": device, "monthly_fee": fee, "churned": 0,
    "payment_method": pay, "number_of_profiles": profiles,
    "avg_watch_time_per_day": daily, "favorite_genre": genre,
}])
eng = add_engineered(pd.concat([raw, sim], ignore_index=True)).tail(1)
X = eng[bundle["num"] + bundle["cat"]]
p = float(bundle["model"].predict_proba(X)[0, 1])

tier = tier_of(p)
rar = p * fee * 12
pop_mean = float(scored["churn_probability"].mean())

g1, g2 = st.columns([1.2, 1])
with g1:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=p * 100,
        number={"suffix": "%"},
        delta={"reference": pop_mean * 100},
        title={"text": "Predicted churn probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": RISK_COLORS[tier]},
            "steps": [
                {"range": [0, RISK_LOW_MAX * 100], "color": "rgba(46,158,91,.30)"},
                {"range": [RISK_LOW_MAX * 100, RISK_HIGH_MIN * 100], "color": "rgba(224,168,0,.30)"},
                {"range": [RISK_HIGH_MIN * 100, 100], "color": "rgba(229,9,20,.30)"},
            ],
            "threshold": {"line": {"color": "#F5F5F5", "width": 3}, "value": pop_mean * 100},
        },
    ))
    gauge.update_layout(height=330, margin=dict(t=60, b=10))
    st.plotly_chart(gauge, use_container_width=True)
    st.caption(f"Delta / white line = population average ({pop_mean * 100:.0f}%). "
               f"Bands = risk tiers (Low ≤ {RISK_LOW_MAX:.0%} · High ≥ {RISK_HIGH_MIN:.0%}).")
with g2:
    st.metric("Risk tier", tier)
    st.metric("Revenue at risk (annual)", fmt_money(rar),
              help="churn probability × monthly fee × 12")
    st.metric("Derived segment / recency",
              f"{eng['engagement_segment'].iloc[0]} / {eng['recency_bucket'].iloc[0]}",
              help="Engineered exactly as in training (population tertiles / recency bands).")
    st.markdown(f"**Recommended action:** {ACTIONS[tier]}")

# where this customer sits in the population
hist = px.histogram(scored, x="churn_probability", nbins=40,
                    title="Where this customer sits in the scored population",
                    color_discrete_sequence=[NETFLIX_RED])
hist.add_vline(x=p, line_width=3, line_dash="dash", line_color="#F5F5F5",
               annotation_text="simulated customer", annotation_position="top")
hist.update_layout(height=300, margin=dict(t=46, b=10), xaxis_title="Churn probability",
                   yaxis_title="Customers", showlegend=False)
st.plotly_chart(hist, use_container_width=True)
st.caption("Probabilities cluster near 0 and 1 because the synthetic data is highly "
           "separable (see the honesty checkpoint) — try lowering watch time or raising "
           "days-since-login to watch the prediction respond. The borderline demo cases "
           "(🌗 ⚠️ 💤) sit in the rare mid-range on purpose.")
