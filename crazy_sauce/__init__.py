"""
Utilities for predicting whether a Crazy Schnitzel basket also contains Crazy Sauce.

This package contains data preparation helpers, a lightweight logistic regression
implementation, and evaluation utilities used by main.py.
"""

from .constants import ANCHOR_PRODUCT, SAUCE_NAMES, TARGET_SAUCE
from .data_prep import (
    build_all_sauce_dataset,
    build_feature_matrix,
    build_product_matrix,
    make_train_test_split,
)
from .logreg import LogisticRegressionGD
from .metrics import compute_classification_metrics, summarize_coefficients

__all__ = [
    "ANCHOR_PRODUCT",
    "SAUCE_NAMES",
    "TARGET_SAUCE",
    "build_all_sauce_dataset",
    "build_feature_matrix",
    "build_product_matrix",
    "make_train_test_split",
    "LogisticRegressionGD",
    "compute_classification_metrics",
    "summarize_coefficients",
]
