
import sys
import os
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(os.path.abspath(".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from crazy_sauce import (
    ANCHOR_PRODUCT,
    TARGET_SAUCE,
    SAUCE_NAMES,
    LogisticRegressionGD,
    build_feature_matrix,
    build_all_sauce_dataset,
    build_product_matrix,
    make_train_test_split,
)
from crazy_sauce.recommendation import train_per_sauce_models
from crazy_sauce.ranking import (
    BernoulliNaiveBayesUpsell,
    make_partial_baskets,
    evaluate_ranking,
)

# Configuration
CSV_PATH = Path("../data/ap_dataset.csv")
TEST_SIZE = 0.2
SPLIT_MODE = "temporal"
RANDOM_STATE = 42

def plot_confusion_matrix_and_roc(y_test, y_probs, y_pred, filename_cm, filename_roc):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {TARGET_SAUCE}")
    plt.tight_layout()
    plt.savefig(filename_cm)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {TARGET_SAUCE}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename_roc)
    plt.close()

def run_task_2_1():
    print("Generating plots for Task 2.1 (Crazy Sauce)...")
    X, y, receipt_time, meta = build_feature_matrix(
        CSV_PATH,
        anchor_product=ANCHOR_PRODUCT,
        target_sauce=TARGET_SAUCE,
        drop_target_feature=True,
        binary_product_counts=False,
    )
    
    train_ids, test_ids = make_train_test_split(
        X, y, receipt_time, test_size=TEST_SIZE, mode=SPLIT_MODE
    )
    X_train, X_test = X.loc[train_ids], X.loc[test_ids]
    y_train, y_test = y.loc[train_ids], y.loc[test_ids]
    
    # Custom GD Model
    model = LogisticRegressionGD(lr=0.1, max_iter=4000, tol=1e-6, l2=0.01)
    model.fit(X_train.values, y_train.values)
    probs = model.predict_proba(X_test.values)
    preds = (probs >= 0.5).astype(int)
    
    plot_confusion_matrix_and_roc(
        y_test, probs, preds, 
        "confusion_matrix_crazy.png", 
        "roc_curve_crazy.png"
    )

def run_task_2_2():
    print("Generating plots for Task 2.2 (All Sauces ROC)...")
    X, sauce_labels, receipt_time, meta = build_all_sauce_dataset(
        CSV_PATH,
        drop_sauce_features=True,
        binary_product_counts=False,
    )
    
    stratify_any_sauce = (sauce_labels.sum(axis=1) > 0).astype(int)
    train_ids, test_ids = make_train_test_split(
        X,
        stratify_any_sauce,
        receipt_time,
        test_size=TEST_SIZE,
        mode=SPLIT_MODE,
        stratify_series=stratify_any_sauce,
    )
    
    # We need the ground truth for test set
    y_test_all = sauce_labels.loc[test_ids]
    
    # Train models and get probs
    _, proba_test = train_per_sauce_models(
        X, sauce_labels, train_ids, test_ids, max_iter=4000
    )
    
    plt.figure(figsize=(10, 8))
    for sauce in SAUCE_NAMES:
        if sauce not in proba_test.columns:
            continue
        y_true = y_test_all[sauce]
        y_score = proba_test[sauce]
        
        # Skip if single class
        if y_true.nunique() < 2:
            continue
            
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{sauce} (AUC = {roc_auc:.3f})")
        
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Sauce")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curves_per_sauce.png")
    plt.close()

def run_task_ranking():
    print("Generating plots for Ranking (Hit@K)...")
    top_ks = [1, 3, 5]
    X, receipt_time, price_map, _ = build_product_matrix(
        CSV_PATH, binary_product_counts=True, drop_zero_variance=True
    )
    
    stratify_series = (X.sum(axis=1) > 0).astype(int)
    train_ids, test_ids = make_train_test_split(
        X,
        stratify_series,
        receipt_time,
        test_size=TEST_SIZE,
        mode=SPLIT_MODE,
        stratify_series=stratify_series,
    )
    X_train, X_test = X.loc[train_ids], X.loc[test_ids]
    
    candidate_counts = X_train.sum()
    candidate_products = [
        p for p, cnt in candidate_counts.items() if cnt >= 20
    ]
    
    nb_model = BernoulliNaiveBayesUpsell(alpha=1.0)
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
        random_state=RANDOM_STATE,
    )
    
    summary = evaluate_ranking(
        model=nb_model,
        scenarios=scenarios,
        candidate_products=candidate_products,
        price_map=price_map,
        popularity_order=popularity_order,
        revenue_order=revenue_order,
        top_ks=top_ks,
    )
    
    # Plotting Hit@K
    models = ["Model", "Popularity", "Revenue"]
    x = np.arange(len(top_ks))
    width = 0.25
    
    plt.figure(figsize=(8, 6))
    
    # Data structure: summary['model'][k], summary['popularity'][k]...
    vals_model = [summary["model"][k] for k in top_ks]
    vals_pop = [summary["popularity"][k] for k in top_ks]
    vals_rev = [summary["revenue"][k] for k in top_ks]
    
    plt.bar(x - width, vals_model, width, label="Model (NB + Price)")
    plt.bar(x, vals_pop, width, label="Popularity")
    plt.bar(x + width, vals_rev, width, label="Revenue")
    
    plt.ylabel("Hit@K Score")
    plt.xlabel("K")
    plt.title("Hit@K Performance Comparison")
    plt.xticks(x, [f"K={k}" for k in top_ks])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("ranking_hit_k.png")
    plt.close()

if __name__ == "__main__":
    try:
        run_task_2_1()
        run_task_2_2()
        run_task_ranking()
        print("All plots generated successfully.")
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
