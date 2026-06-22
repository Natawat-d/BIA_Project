"""Model — mandatory naive baselines.

A model must beat these to be worth anything. On balanced data the majority-class
accuracy (~50%) is itself the argument for not using accuracy as the headline.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


def majority_class(y_train, y_test) -> dict:
    const = int(round(float(np.mean(y_train))))  # most frequent class
    pred = np.full(len(y_test), const)
    proba = np.full(len(y_test), float(np.mean(y_train)))
    return {
        "name": "Majority class",
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "roc_auc": 0.5,
    }


def inactivity_heuristic(train_df, test_df, target, col: str = "last_login_days") -> dict:
    """'Inactive >= T days -> churn'. T chosen on train to maximise F1."""
    thresholds = np.quantile(train_df[col], np.linspace(0.1, 0.9, 17))
    best_t, best_f1 = thresholds[0], -1.0
    for t in thresholds:
        pred = (train_df[col] >= t).astype(int)
        s = f1_score(train_df[target], pred, zero_division=0)
        if s > best_f1:
            best_f1, best_t = s, t
    pred = (test_df[col] >= best_t).astype(int)
    denom = max(float(test_df[col].max()), 1.0)
    proba = (test_df[col] / denom).clip(0, 1)
    return {
        "name": f"Inactive >= {best_t:.0f} days",
        "accuracy": float(accuracy_score(test_df[target], pred)),
        "f1": float(f1_score(test_df[target], pred, zero_division=0)),
        "pr_auc": float(average_precision_score(test_df[target], proba)),
        "roc_auc": float(roc_auc_score(test_df[target], proba)),
    }
