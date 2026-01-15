from __future__ import annotations

"""
CLI entrypoint for the sauce and upsell experiments. Pick experiment via
--experiment: crazy_sauce (2.1), all_sauces (2.2), ranking (upsell Hit@K).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from crazy_sauce import (
    ANCHOR_PRODUCT,
    SAUCE_NAMES,
    TARGET_SAUCE,
    LogisticRegressionGD,
    build_all_sauce_dataset,
    build_feature_matrix,
    build_product_matrix,
    compute_classification_metrics,
    make_train_test_split,
    summarize_coefficients,
)
from crazy_sauce.metrics import majority_baseline
from crazy_sauce.recommendation import evaluate_recommendations, train_per_sauce_models
from crazy_sauce.ranking import (
    BernoulliNaiveBayesUpsell,
    evaluate_ranking,
    make_partial_baskets,
)


def describe_features(meta: dict):
    """Print a short summary of dataset size, balance, and dates."""
    print(f"Total baskets (anchor={ANCHOR_PRODUCT}): {meta['num_receipts']}")
    print(f"Positive rate for {TARGET_SAUCE}: {meta['positive_rate']:.3f}")
    print(
        f"Period: {meta['date_range'][0]} -> {meta['date_range'][1]}, features: {meta['feature_count']}"
    )
    if meta["dropped_zero_variance"]:
        print(f"Dropped zero-variance columns: {meta['dropped_zero_variance']}")


def print_metrics(label: str, metrics: dict):
    """Nicely format the metric dict produced by compute_classification_metrics."""
    cm = metrics["confusion_matrix"]
    print(
        f"[{label}] Acc {metrics['accuracy']:.3f} | "
        f"Prec {metrics['precision']:.3f} | Rec {metrics['recall']:.3f} | "
        f"F1 {metrics['f1']:.3f} | ROC-AUC {metrics['roc_auc']:.3f}"
    )
    print(f"    Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}")


def build_arg_parser():
    """CLI parser with knobs for splits, model params, and experiment choice."""
    parser = argparse.ArgumentParser(
        description="Predict if a Crazy Schnitzel basket will include Crazy Sauce."
    )
    parser.add_argument("--csv-path", type=Path, default=Path("data/ap_dataset.csv"))
    parser.add_argument(
        "--experiment",
        choices=["crazy_sauce", "all_sauces", "ranking"],
        default="crazy_sauce",
        help="crazy_sauce: task 2.1; all_sauces: task 2.2 per-sauce models + recommendations.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["temporal", "random"],
        default="temporal",
        help="Temporal split keeps later receipts for test.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--binary-counts", action="store_true", help="Use 0/1 product indicators.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for custom logistic GD.")
    parser.add_argument("--l2", type=float, default=0.01, help="L2 regularization for custom GD.")
    parser.add_argument("--max-iter", type=int, default=4000, help="Max steps for custom GD.")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for early stop in GD.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K for sauce recommendation eval.")
    parser.add_argument(
        "--ranking-topk",
        type=str,
        default="1,3,5",
        help="Comma-separated K values for ranking evaluation.",
    )
    parser.add_argument(
        "--ranking-min-occ",
        type=int,
        default=20,
        help="Minimum train occurrences for a product to be a ranking candidate.",
    )
    parser.add_argument(
        "--ranking-alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing alpha for custom Naive Bayes ranking model.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splits and partial basket sampling.",
    )
    return parser


def adjust_sklearn_coefficients(pipeline, feature_names: list[str]) -> tuple[float, np.ndarray]:
    """
    Convert coefficients from standardized space back to original units.
    """
    logreg = pipeline.named_steps["logisticregression"]
    scaler: StandardScaler = pipeline.named_steps["standardscaler"]

    scaled_coef = logreg.coef_[0]
    raw_coef = scaled_coef / scaler.scale_
    intercept = logreg.intercept_[0] - np.sum((scaler.mean_ / scaler.scale_) * scaled_coef)
    return intercept, raw_coef


def run_crazy_sauce(args: argparse.Namespace):
    """Task 2.1: Crazy Schnitzel baskets → predict Crazy Sauce (custom + sklearn)."""
    X, y, receipt_time, meta = build_feature_matrix(
        args.csv_path,
        anchor_product=ANCHOR_PRODUCT,
        target_sauce=TARGET_SAUCE,
        drop_target_feature=True,
        binary_product_counts=args.binary_counts,
    )

    describe_features(meta)

    train_ids, test_ids = make_train_test_split(
        X, y, receipt_time, test_size=args.test_size, mode=args.split_mode
    )
    X_train, X_test = X.loc[train_ids], X.loc[test_ids]
    y_train, y_test = y.loc[train_ids], y.loc[test_ids]

    print(f"Train size: {len(train_ids)}, Test size: {len(test_ids)}")

    baseline = majority_baseline(y_train, y_test)
    print_metrics("Majority baseline", baseline)

    # Scikit-learn logistic regression (reference)
    sk_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    sk_model.fit(X_train, y_train)
    sk_probs = sk_model.predict_proba(X_test)[:, 1]
    sk_metrics = compute_classification_metrics(y_test, sk_probs)
    print_metrics("sklearn LogisticRegression", sk_metrics)

    # Custom logistic regression with gradient descent
    gd_model = LogisticRegressionGD(
        lr=args.lr,
        max_iter=args.max_iter,
        tol=args.tol,
        l2=args.l2,
        verbose=False,
    )
    gd_model.fit(X_train.values, y_train.values)
    gd_probs = gd_model.predict_proba(X_test.values)
    gd_metrics = compute_classification_metrics(y_test, gd_probs)
    print_metrics("Custom GD logistic", gd_metrics)
    print(f"    GD steps: {gd_model.n_iter_}")

    gd_top = summarize_coefficients(gd_model.coef_, list(X_train.columns), top_k=8)
    sk_intercept, sk_coef = adjust_sklearn_coefficients(sk_model, list(X_train.columns))
    sk_top = summarize_coefficients(sk_coef, list(X_train.columns), top_k=8)

    print("\nTop positive features (custom GD):")
    print(gd_top["positive"])
    print("\nTop negative features (custom GD):")
    print(gd_top["negative"])

    print("\nTop positive features (sklearn):")
    print(sk_top["positive"])
    print("\nTop negative features (sklearn):")
    print(sk_top["negative"])

    print(f"\nScikit intercept (raw units): {sk_intercept:.4f}")
    print(f"Custom intercept (standardized space): {gd_model.intercept_:.4f}")


def run_all_sauces(args: argparse.Namespace):
    """Task 2.2: one logistic model per sauce + Top-K recommendation eval."""
    X, sauce_labels, receipt_time, meta = build_all_sauce_dataset(
        args.csv_path,
        drop_sauce_features=True,
        binary_product_counts=args.binary_counts,
    )

    print(
        f"All receipts: {meta['num_receipts']}, features: {meta['feature_count']}, period: {meta['date_range'][0]} -> {meta['date_range'][1]}"
    )
    print("Positive rate per sauce:")
    for sauce, rate in meta["positive_rates"].items():
        print(f"  {sauce}: {rate:.3f}")
    if meta["dropped_zero_variance"]:
        print(f"Dropped zero-variance columns: {meta['dropped_zero_variance']}")

    # For random split, stratify by "any sauce in basket" to avoid degenerate splits
    stratify_any_sauce = (sauce_labels.sum(axis=1) > 0).astype(int)
    train_ids, test_ids = make_train_test_split(
        X,
        stratify_any_sauce,
        receipt_time,
        test_size=args.test_size,
        mode=args.split_mode,
        stratify_series=stratify_any_sauce,
    )
    X_train, X_test = X.loc[train_ids], X.loc[test_ids]

    print(f"Train baskets: {len(train_ids)}, Test baskets: {len(test_ids)}")

    metrics_per_sauce, proba_test = train_per_sauce_models(
        X, sauce_labels, train_ids, test_ids, max_iter=args.max_iter
    )

    popularity_order = (
        sauce_labels.loc[train_ids].sum().sort_values(ascending=False).index.tolist()
    )
    recomm_eval = evaluate_recommendations(
        proba_df=proba_test,
        sauce_labels=sauce_labels,
        test_ids=test_ids,
        popularity_topk=popularity_order,
        k=args.top_k,
    )

    print("\nPer-sauce logistic regression metrics (test set)")
    for sauce, m in metrics_per_sauce.items():
        print(
            f"{sauce}: Acc {m['accuracy']:.3f}, Prec {m['precision']:.3f}, Rec {m['recall']:.3f}, F1 {m['f1']:.3f}, ROC-AUC {m['roc_auc']:.3f}"
        )

    print(f"\nRecommendation evaluation (Top-{args.top_k})")
    print(
        f"Hit@{args.top_k}: {recomm_eval['hit_at_k']:.3f} vs baseline {recomm_eval['baseline_hit_at_k']:.3f}"
    )
    print(
        f"Precision@{args.top_k}: {recomm_eval['precision_at_k']:.3f} vs baseline {recomm_eval['baseline_precision_at_k']:.3f}"
    )
    print(f"Evaluated baskets: {recomm_eval['considered_baskets']}")
    print(f"Baseline popularity order: {popularity_order}")


def run_ranking(args: argparse.Namespace):
    """Upsell ranking with Bernoulli NB scored by probability × price."""
    top_ks = [int(k.strip()) for k in args.ranking_topk.split(",") if k.strip()]

    X, receipt_time, price_map, meta = build_product_matrix(
        args.csv_path, binary_product_counts=True, drop_zero_variance=True
    )
    print(
        f"All receipts: {meta['num_receipts']}, features: {meta['feature_count']}, period: {meta['date_range'][0]} -> {meta['date_range'][1]}"
    )
    if meta["dropped_zero_variance"]:
        print(f"Dropped zero-variance columns: {meta['dropped_zero_variance']}")

    # Split train/test at receipt level
    stratify_series = (X.sum(axis=1) > 0).astype(int)
    train_ids, test_ids = make_train_test_split(
        X,
        stratify_series,
        receipt_time,
        test_size=args.test_size,
        mode=args.split_mode,
        stratify_series=stratify_series,
    )
    X_train, X_test = X.loc[train_ids], X.loc[test_ids]

    # Candidate products: frequent enough in train
    candidate_counts = X_train.sum()
    candidate_products = [
        p for p, cnt in candidate_counts.items() if cnt >= args.ranking_min_occ
    ]
    print(f"Candidate products (>= {args.ranking_min_occ} occurrences): {len(candidate_products)}")

    nb_model = BernoulliNaiveBayesUpsell(alpha=args.ranking_alpha)
    nb_model.fit(X_train[candidate_products])

    popularity_order = (
        candidate_counts[candidate_products].sort_values(ascending=False).index.tolist()
    )
    revenue_series = (
        candidate_counts[candidate_products]
        * pd.Series({p: price_map.get(p, 0.0) for p in candidate_products})
    )
    revenue_order = revenue_series.sort_values(ascending=False).index.tolist()

    scenarios = make_partial_baskets(
        X_test[candidate_products],
        candidate_products,
        random_state=args.random_state,
    )
    print(f"Evaluating {len(scenarios)} partial baskets.")

    ranking_summary = evaluate_ranking(
        model=nb_model,
        scenarios=scenarios,
        candidate_products=candidate_products,
        price_map=price_map,
        popularity_order=popularity_order,
        revenue_order=revenue_order,
        top_ks=top_ks,
    )

    print("\nRanking Hit@K (value = fraction of times removed product is recovered):")
    for k in top_ks:
        print(
            f"K={k}: Model {ranking_summary['model'][k]:.3f}, "
            f"Popularity {ranking_summary['popularity'][k]:.3f}, "
            f"Revenue {ranking_summary['revenue'][k]:.3f}"
        )
    print(f"Evaluated baskets: {ranking_summary['evaluated']}")


def main(args: argparse.Namespace | None = None):
    """Dispatch to the selected experiment."""
    args = args or build_arg_parser().parse_args()

    if args.experiment == "crazy_sauce":
        run_crazy_sauce(args)
    elif args.experiment == "all_sauces":
        run_all_sauces(args)
    else:
        run_ranking(args)


if __name__ == "__main__":
    main()
