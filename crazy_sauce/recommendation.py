from __future__ import annotations

"""
Helpers for task 2.2: train one logistic model per sauce and evaluate
recommendation quality (Hit@K/Precision@K) against a popularity baseline.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .constants import SAUCE_NAMES
from .metrics import compute_classification_metrics


def train_per_sauce_models(
    X: pd.DataFrame,
    sauce_labels: pd.DataFrame,
    train_ids: pd.Index,
    test_ids: pd.Index,
    max_iter: int = 4000,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """
    Train one LogisticRegression per sauce; returns metrics and test probabilities.
    Each sauce feature is dropped from X to avoid trivial leakage.
    """
    metrics_per_sauce: dict[str, dict] = {}
    proba_test = pd.DataFrame(index=test_ids)

    for sauce in SAUCE_NAMES:
        y = sauce_labels[sauce]
        # Drop the current sauce feature if present to prevent leakage
        feature_cols = X.columns.drop(sauce) if sauce in X.columns else X.columns
        X_train = X.loc[train_ids, feature_cols]
        X_test = X.loc[test_ids, feature_cols]

        y_train = y.loc[train_ids]
        y_test = y.loc[test_ids]

        if y_train.nunique() < 2:
            probs_test = np.full(len(test_ids), float(y_train.iloc[0]))
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter))
            model.fit(X_train, y_train)
            probs_test = model.predict_proba(X_test)[:, 1]

        proba_test[sauce] = probs_test
        metrics_per_sauce[sauce] = compute_classification_metrics(y_test, probs_test)

    return metrics_per_sauce, proba_test


def evaluate_recommendations(
    proba_df: pd.DataFrame,
    sauce_labels: pd.DataFrame,
    test_ids: pd.Index,
    popularity_topk: list[str],
    k: int = 3,
):
    """
    Evaluate Top-K recommendation when sauces are ranked by predicted probability.
    """
    hits, precisions = [], []
    base_hits, base_precisions = [], []
    considered = 0

    for bon_id in test_ids:
        actual = {s for s in SAUCE_NAMES if sauce_labels.at[bon_id, s] == 1}
        if not actual:
            continue  # basket without sauce, not counted for hit@k
        considered += 1

        topk_pred = proba_df.loc[bon_id].nlargest(k).index
        hits.append(1 if actual & set(topk_pred) else 0)
        precisions.append(len(actual & set(topk_pred)) / k)

        topk_base = popularity_topk[:k]
        base_hits.append(1 if actual & set(topk_base) else 0)
        base_precisions.append(len(actual & set(topk_base)) / k)

    summary = {
        "considered_baskets": considered,
        "hit_at_k": float(np.mean(hits)) if hits else float("nan"),
        "precision_at_k": float(np.mean(precisions)) if precisions else float("nan"),
        "baseline_hit_at_k": float(np.mean(base_hits)) if base_hits else float("nan"),
        "baseline_precision_at_k": float(np.mean(base_precisions)) if base_precisions else float("nan"),
    }
    return summary
