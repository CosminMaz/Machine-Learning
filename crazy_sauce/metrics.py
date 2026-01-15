from __future__ import annotations

"""
Metric helpers used across experiments (classification summaries and coef dumps).
"""

import numpy as np
import pandas as pd
from sklearn import metrics


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series, probs: np.ndarray, threshold: float = 0.5
):
    """Compute standard binary metrics given probabilities and a threshold."""
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    try:
        roc_auc = metrics.roc_auc_score(y_true, probs)
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": metrics.accuracy_score(y_true, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": metrics.confusion_matrix(y_true, preds, labels=[0, 1]),
    }


def majority_baseline(y_train: np.ndarray | pd.Series, y_test: np.ndarray | pd.Series):
    """
    Predicts the majority class probability learned from the training set.
    """
    prob = float(np.mean(y_train))
    probs = np.full_like(y_test, prob, dtype=float)
    return compute_classification_metrics(y_test, probs)


def summarize_coefficients(
    coef: np.ndarray, feature_names: list[str], top_k: int = 8
) -> dict[str, pd.Series]:
    coef_series = pd.Series(coef, index=feature_names)
    coef_sorted = coef_series.sort_values()
    return {
        "positive": coef_sorted.tail(top_k)[::-1],
        "negative": coef_sorted.head(top_k),
    }
