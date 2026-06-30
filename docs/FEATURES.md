# Model Features — Netflix Retention DSS

What the churn models are trained on, why, and how it's processed. Source of
truth: [`src/config.py`](src/config.py) (feature lists) and
[`src/prep/features.py`](src/prep/features.py) (engineering + split).

**Summary:** **15 features** (7 numeric + 8 categorical) → **~35 model inputs**
after one-hot encoding. The identifier and the label are excluded.

---

## 1. Numeric features (7)

| Feature | Type | Source | Description |
|---|---|---|---|
| `age` | numeric | raw | Subscriber age (years) |
| `watch_hours` | numeric | raw | Total hours watched |
| `last_login_days` | numeric | raw | Days since last login (recency) |
| `monthly_fee` | numeric | raw | Monthly subscription fee ($) |
| `number_of_profiles` | numeric | raw | Profiles on the account |
| `avg_watch_time_per_day` | numeric | raw | Average daily watch time |
| `watch_per_profile` | numeric | **engineered** | `watch_hours / number_of_profiles` (divide-by-zero → falls back to `watch_hours`) |

## 2. Categorical features (8)

| Feature | Type | Source | Values |
|---|---|---|---|
| `gender` | categorical | raw | Male / Female / … |
| `subscription_type` | categorical | raw | Basic / Standard / Premium |
| `region` | categorical | raw | 6 continents |
| `device` | categorical | raw | Mobile / TV / Laptop / … |
| `payment_method` | categorical | raw | Credit card / PayPal / … |
| `favorite_genre` | categorical | raw | Drama / Action / … |
| `engagement_segment` | categorical | **engineered** | Tertiles of `avg_watch_time_per_day` → **Low / Medium / High** |
| `recency_bucket` | categorical | **engineered** | From `last_login_days`: **Active** (≤7d) · **Lapsing** (≤30d) · **Dormant** (>30d) |

## 3. Excluded columns

| Column | Why excluded |
|---|---|
| `customer_id` | Identifier — no predictive value (`ID_COL` in config) |
| `churned` | The **target label** (0 = retained, 1 = churned), not a feature |

---

## 4. Engineered features (definitions)

Created in `add_engineered()` — used by **both** the models and the dashboard so
the two always agree:

- **`watch_per_profile`** = `watch_hours / number_of_profiles` — normalises viewing
  by household size (heavy account vs. one heavy viewer).
- **`engagement_segment`** = `pd.qcut(avg_watch_time_per_day, 3)` → Low / Medium /
  High — a behavioural tier used across the descriptive dashboard.
- **`recency_bucket`** = `pd.cut(last_login_days, [−1, 7, 30, ∞])` → Active / Lapsing
  / Dormant — a human-readable recency band.

---

## 5. Preprocessing (leak-free)

All preprocessing is wrapped in a scikit-learn **`Pipeline`**, so it is fit on the
**training fold only** — no information leaks from test to train
([`build_preprocessor`](src/prep/features.py)):

```
ColumnTransformer(
    ("num", StandardScaler(),               numeric_features),   # scale for LR
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
)
```

- **StandardScaler** on numerics — required by Logistic Regression (tree models are
  scale-invariant, but the shared pipeline keeps it consistent).
- **OneHotEncoder** on categoricals — expands the 8 categoricals into indicator
  columns; `handle_unknown="ignore"` guards against unseen categories at inference.
- **Net effect:** 15 features → **~35 encoded inputs** (verified: the fitted
  Logistic Regression has 35 coefficients).

Training loop (per model): `Pipeline(preprocessor → estimator)`, stratified
k-fold CV for model selection, then fit on train and evaluate on the held-out
test set — see [`src/model/train.py`](src/model/train.py).

---

## 6. Leakage audit

`LEAKAGE_SUSPECTS = ["last_login_days", "recency_bucket"]`

Recency is almost a proxy for churn (a churned user stops logging in), so the audit
**drops both** and re-fits:

| Variant | Test PR-AUC |
|---|---|
| With recency features | 1.000 |
| Without recency features | 0.997 |
| **Delta** | **0.003** |

Removing recency barely moves the score → recency is **not** the main driver of the
near-perfect result. Note `recency_bucket` is derived **from** `last_login_days`, so
dropping both tests that whole pathway.

> **Open item (threats to validity):** `engagement_segment` is derived from
> `avg_watch_time_per_day`. If low watch-time is partly a *consequence* of churning
> (a post-churn artifact) rather than a cause, that is the most likely remaining
> leak. Next step: audit every feature for post-churn artifacts, not just recency.

---

## 7. Where this lives in code

| Concern | File |
|---|---|
| Feature lists, ID, target, leakage suspects | `src/config.py` |
| Feature engineering + `X/y` split | `src/prep/features.py` |
| Preprocessing pipeline | `src/prep/features.py` (`build_preprocessor`) |
| Training / CV / leakage audit | `src/model/train.py` |

**One-line summary:** *15 features — 6 raw numeric, 6 raw categorical, plus 3
engineered (watch-per-profile, engagement segment, recency bucket) — one-hot encoded
to ~35 inputs; identifier and label excluded; all preprocessing fit inside the
pipeline on training data only.*
