"""Data Preparation — stratified train/test split."""
from __future__ import annotations

from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE


def stratified_split(X, y):
    """Stratified split on the churn label with a fixed seed for reproducibility."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
