"""Model — train baselines + LR/DT/RF/XGBoost, evaluate, calibrate, audit leakage.

Run with:  python -m src.model.train
Writes the calibrated best model to models/ and a full report to reports/metrics.json.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CLEAN_CSV,
    CV_FOLDS,
    LEAKAGE_SUSPECTS,
    METRICS_JSON,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET,
)
from src.model import baselines
from src.model.evaluate import score_metrics
from src.prep.features import add_engineered, build_preprocessor, split_X_y
from src.prep.split import stratified_split

log = logging.getLogger(__name__)


def get_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("XGBoost unavailable (%s) — continuing without it.", exc)
    return models


def _cv_metrics(pipe, X, y) -> dict:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    return score_metrics(y, proba)


def _importances(pipe, num, cat) -> list[dict]:
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    names = list(num)
    ohe = prep.named_transformers_["cat"]
    names += list(ohe.get_feature_names_out(cat))
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        imp = np.abs(np.asarray(clf.coef_[0], dtype=float))
    else:
        return []
    pairs = sorted(zip(names, imp), key=lambda kv: kv[1], reverse=True)[:15]
    return [{"feature": n, "importance": float(v)} for n, v in pairs]


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = add_engineered(pd.read_csv(CLEAN_CSV))

    X, y, num, cat = split_X_y(df)
    X_tr, X_te, y_tr, y_te = stratified_split(X, y)

    results, fitted = {}, {}
    for name, clf in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor(num, cat)), ("clf", clf)])
        cvm = _cv_metrics(pipe, X_tr, y_tr)
        pipe.fit(X_tr, y_tr)
        tem = score_metrics(y_te, pipe.predict_proba(X_te)[:, 1])
        results[name] = {"cv": cvm, "test": tem}
        fitted[name] = pipe
        log.info(
            "%-20s | CV PR-AUC=%.3f ROC-AUC=%.3f | Test PR-AUC=%.3f F1=%.3f Acc=%.3f",
            name, cvm["pr_auc"], cvm["roc_auc"], tem["pr_auc"], tem["f1"], tem["accuracy"],
        )

    # naive baselines
    train_df, test_df = df.loc[X_tr.index], df.loc[X_te.index]
    base = [
        baselines.majority_class(y_tr, y_te),
        baselines.inactivity_heuristic(train_df, test_df, TARGET),
    ]
    for b in base:
        log.info("Baseline %-22s | Acc=%.3f F1=%.3f PR-AUC=%.3f",
                 b["name"], b["accuracy"], b["f1"], b["pr_auc"])

    # select best by CV PR-AUC, then calibrate
    best_name = max(results, key=lambda k: results[k]["cv"]["pr_auc"])
    log.info("Best model by CV PR-AUC: %s", best_name)
    calibrated = CalibratedClassifierCV(fitted[best_name], method="isotonic", cv=5)
    calibrated.fit(X_tr, y_tr)
    cal_metrics = score_metrics(y_te, calibrated.predict_proba(X_te)[:, 1])
    log.info("Calibrated %s | PR-AUC=%.3f Brier=%.3f", best_name,
             cal_metrics["pr_auc"], cal_metrics["brier"])

    # leakage audit: retrain best WITHOUT recency suspects
    Xn, yn, numn, catn = split_X_y(df, drop_features=LEAKAGE_SUSPECTS)
    Xn_tr, Xn_te, yn_tr, yn_te = stratified_split(Xn, yn)
    pipe_n = Pipeline([("prep", build_preprocessor(numn, catn)),
                       ("clf", get_models()[best_name])])
    pipe_n.fit(Xn_tr, yn_tr)
    leak = {
        "suspects": LEAKAGE_SUSPECTS,
        "with_suspects": results[best_name]["test"],
        "without_suspects": score_metrics(yn_te, pipe_n.predict_proba(Xn_te)[:, 1]),
    }
    log.info("Leakage audit | PR-AUC with=%.3f without=%.3f (delta=%.3f)",
             leak["with_suspects"]["pr_auc"], leak["without_suspects"]["pr_auc"],
             leak["with_suspects"]["pr_auc"] - leak["without_suspects"]["pr_auc"])

    # persist best calibrated model + feature lists
    dump({"model": calibrated, "num": num, "cat": cat}, MODEL_PATH)
    log.info("Saved model -> %s", MODEL_PATH)

    report = {
        "dataset": {"rows": int(len(df)), "churn_rate": float(y.mean())},
        "best_model": best_name,
        "models": results,
        "baselines": base,
        "calibrated_test": cal_metrics,
        "leakage_audit": leak,
        "drivers": _importances(fitted[best_name], num, cat),
    }
    METRICS_JSON.write_text(json.dumps(report, indent=2))
    log.info("Wrote metrics -> %s", METRICS_JSON)


if __name__ == "__main__":
    run()
