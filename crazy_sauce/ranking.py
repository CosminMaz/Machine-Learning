from __future__ import annotations

"""
Lightweight upsell ranking utilities: a custom Bernoulli Naive Bayes that scores
products for a partial basket and evaluation helpers for Hit@K.
"""

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def logsumexp(log_vals: np.ndarray) -> float:
    m = np.max(log_vals)
    return m + np.log(np.sum(np.exp(log_vals - m)))


class BernoulliNaiveBayesUpsell:
    """
    Simple Bernoulli Naive Bayes to score P(product | cart). It only looks at
    binary presence/absence and uses Laplace smoothing to stay stable on rares.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.feature_names: list[str] = []
        self.priors_: np.ndarray | None = None
        self.cond_probs_: np.ndarray | None = None  # shape [targets, features]

    def fit(self, X: pd.DataFrame):
        """
        Fit the model on a basket x product indicator matrix.
        """
        X_arr = X.values.astype(int)
        n_samples, n_features = X_arr.shape
        self.feature_names = list(X.columns)

        # Prior P(t = 1)
        target_counts = X_arr.sum(axis=0)
        self.priors_ = (target_counts + self.alpha) / (
            n_samples + 2 * self.alpha
        )

        cond_probs = np.zeros((n_features, n_features))
        for t in range(n_features):
            mask = X_arr[:, t] == 1
            if not mask.any():
                cond_probs[t, :] = 0.5  # neutral when no positives exist
                continue
            X_pos = X_arr[mask]
            counts = X_pos.sum(axis=0)
            cond_probs[t, :] = (counts + self.alpha) / (
                len(X_pos) + 2 * self.alpha
            )
        self.cond_probs_ = cond_probs
        return self

    def _log_joint(self, x_vec: np.ndarray, target_idx: int) -> float:
        """
        Compute log P(target, features) for a single basket vector.
        """
        p_t = self.priors_[target_idx]
        logp = math.log(p_t)
        cp = self.cond_probs_[target_idx]
        for j, xj in enumerate(x_vec):
            if j == target_idx:
                continue  # skip target feature to avoid leakage
            prob = cp[j]
            prob = min(max(prob, 1e-9), 1 - 1e-9)
            logp += xj * math.log(prob) + (1 - xj) * math.log(1 - prob)
        return logp

    def predict_proba(
        self,
        x: pd.Series | np.ndarray,
        candidate_products: Sequence[str] | None = None,
    ) -> pd.Series:
        """
        Return normalized probabilities for candidate products given a basket.
        """
        if self.priors_ is None or self.cond_probs_ is None:
            raise RuntimeError("Model not fitted.")
        if isinstance(x, pd.Series):
            x_vec = x.reindex(self.feature_names).fillna(0).to_numpy(dtype=int)
        else:
            x_vec = np.asarray(x, dtype=int)

        indices = (
            [self.feature_names.index(p) for p in candidate_products]
            if candidate_products is not None
            else list(range(len(self.feature_names)))
        )

        log_joints = np.array([self._log_joint(x_vec, idx) for idx in indices])
        norm = logsumexp(log_joints)
        probs = np.exp(log_joints - norm)
        product_names = (
            [self.feature_names[i] for i in indices]
            if candidate_products is None
            else list(candidate_products)
        )
        return pd.Series(probs, index=product_names)


def make_partial_baskets(
    X: pd.DataFrame,
    candidate_products: Iterable[str],
    random_state: int = 42,
):
    """
    Build partial baskets by hiding one random candidate item that is present.
    """
    rng = np.random.default_rng(random_state)
    scenarios = []
    for bon_id, row in X.iterrows():
        present_candidates = [p for p in candidate_products if row.get(p, 0) == 1]
        if not present_candidates:
            continue
        removed = rng.choice(present_candidates)
        partial = row.copy()
        partial[removed] = 0
        scenarios.append({"id_bon": bon_id, "removed": removed, "partial": partial})
    return scenarios


def evaluate_ranking(
    model: BernoulliNaiveBayesUpsell,
    scenarios: list[dict],
    candidate_products: list[str],
    price_map: dict[str, float],
    popularity_order: list[str],
    revenue_order: list[str],
    top_ks: Sequence[int],
):
    """
    Compare ranking hits for the model vs popularity and revenue baselines.
    """
    results = {
        "model": {k: [] for k in top_ks},
        "popularity": {k: [] for k in top_ks},
        "revenue": {k: [] for k in top_ks},
    }

    for sc in scenarios:
        partial = sc["partial"]
        removed = sc["removed"]
        present_set = {p for p in candidate_products if partial.get(p, 0) == 1}

        # Model scores
        probs = model.predict_proba(partial, candidate_products)
        prices = pd.Series({p: price_map.get(p, 0.0) for p in candidate_products})
        scores = probs * prices
        ranked = scores.drop(index=list(present_set), errors="ignore").sort_values(
            ascending=False
        )

        pop_rank = [
            p for p in popularity_order if p not in present_set
        ]
        rev_rank = [
            p for p in revenue_order if p not in present_set
        ]

        for k in top_ks:
            top_model = ranked.head(k).index
            top_pop = pop_rank[:k]
            top_rev = rev_rank[:k]

            results["model"][k].append(1 if removed in top_model else 0)
            results["popularity"][k].append(1 if removed in top_pop else 0)
            results["revenue"][k].append(1 if removed in top_rev else 0)

    summary = {}
    for key in results:
        summary[key] = {
            k: float(np.mean(results[key][k])) if results[key][k] else float("nan")
            for k in top_ks
        }
    summary["evaluated"] = len(scenarios)
    return summary
