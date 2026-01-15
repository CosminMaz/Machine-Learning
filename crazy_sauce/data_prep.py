from __future__ import annotations

"""
Data preparation utilities for the sauce experiments and upsell ranking.
"""

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import ANCHOR_PRODUCT, INTERACTION_PAIRS, SAUCE_NAMES, TARGET_SAUCE


def _clean_name(name: str) -> str:
    """Lightweight normalizer for feature names."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("&", "and")
        .replace("/", "_")
        .replace(",", "")
        .replace(".", "")
    )


def _build_interactions(
    product_counts: pd.DataFrame, pairs: Iterable[tuple[str, str]]
) -> pd.DataFrame:
    """Create simple count-based interaction features for chosen product pairs."""
    features = {}
    for left, right in pairs:
        if left in product_counts.columns and right in product_counts.columns:
            col_name = f"inter_{_clean_name(left)}__{_clean_name(right)}"
            features[col_name] = product_counts[left] * product_counts[right]
    return pd.DataFrame(features, index=product_counts.index)


def build_feature_matrix(
    csv_path: Path,
    anchor_product: str = ANCHOR_PRODUCT,
    target_sauce: str = TARGET_SAUCE,
    drop_target_feature: bool = True,
    binary_product_counts: bool = False,
    interaction_pairs: Sequence[tuple[str, str]] | None = INTERACTION_PAIRS,
):
    """
    Prepare per-receipt features and Crazy Sauce labels.

    The dataset is restricted to receipts containing the anchor product.
    """
    df = pd.read_csv(csv_path)
    df["data_bon"] = pd.to_datetime(df["data_bon"])

    anchor_receipts = df.loc[df["retail_product_name"] == anchor_product, "id_bon"].unique()
    df = df[df["id_bon"].isin(anchor_receipts)].copy()
    df.sort_values(["id_bon", "data_bon"], inplace=True)

    receipt_time = df.groupby("id_bon")["data_bon"].min()
    label_series = (
        df.groupby("id_bon")["retail_product_name"]
        .apply(lambda names: int((names == target_sauce).any()))
        .astype(int)
    )

    grouped = df.groupby("id_bon")
    agg_features = pd.DataFrame(
        {
            "cart_size": grouped.size(),
            "distinct_products": grouped["retail_product_name"].nunique(),
            "total_value": grouped["SalePriceWithVAT"].sum(),
        }
    )

    time_features = pd.DataFrame(
        {
            "day_of_week": receipt_time.dt.dayofweek + 1,
            "hour_of_day": receipt_time.dt.hour,
            "is_weekend": (receipt_time.dt.dayofweek >= 5).astype(int),
        }
    )

    product_counts = (
        df.groupby(["id_bon", "retail_product_name"]).size().unstack(fill_value=0)
    )
    if binary_product_counts:
        product_counts = (product_counts > 0).astype(int)

    if drop_target_feature:
        product_counts = product_counts.drop(columns=[target_sauce], errors="ignore")

    interaction_df = (
        _build_interactions(product_counts, interaction_pairs)
        if interaction_pairs
        else pd.DataFrame(index=product_counts.index)
    )

    X = (
        agg_features.join(time_features)
        .join(product_counts)
        .join(interaction_df)
        .fillna(0)
    )

    zero_var_cols = list(X.columns[X.nunique() <= 1])
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)

    meta = {
        "num_receipts": len(X),
        "positive_rate": float(label_series.mean()),
        "date_range": (receipt_time.min(), receipt_time.max()),
        "feature_count": X.shape[1],
        "dropped_zero_variance": zero_var_cols,
    }

    ordered_index = X.index.sort_values()
    return X.loc[ordered_index], label_series.loc[ordered_index], receipt_time.loc[ordered_index], meta


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    test_size: float = 0.2,
    mode: str = "temporal",
    random_state: int | None = 42,
    stratify_series: pd.Series | None = None,
):
    """Split at basket level, either temporally or randomly (optionally stratified)."""
    if mode == "temporal":
        ordered = timestamps.sort_values().index
        split_at = int(len(ordered) * (1 - test_size))
        train_ids = ordered[:split_at]
        test_ids = ordered[split_at:]
    elif mode == "random":
        stratify_target = stratify_series if stratify_series is not None else y
        train_ids, test_ids = train_test_split(
            X.index, test_size=test_size, random_state=random_state, stratify=stratify_target
        )
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    return train_ids, test_ids


def build_all_sauce_dataset(
    csv_path: Path,
    drop_sauce_features: bool = True,
    binary_product_counts: bool = False,
):
    """
    Build features for every receipt and labels for each sauce.

    By default all sauce product columns are removed from features to align with
    the “cart without sauce yet” recommendation scenario.
    """
    df = pd.read_csv(csv_path)
    df["data_bon"] = pd.to_datetime(df["data_bon"])

    receipt_time = df.groupby("id_bon")["data_bon"].min()
    grouped = df.groupby("id_bon")

    agg_features = pd.DataFrame(
        {
            "cart_size": grouped.size(),
            "distinct_products": grouped["retail_product_name"].nunique(),
            "total_value": grouped["SalePriceWithVAT"].sum(),
        }
    )

    time_features = pd.DataFrame(
        {
            "day_of_week": receipt_time.dt.dayofweek + 1,
            "hour_of_day": receipt_time.dt.hour,
            "is_weekend": (receipt_time.dt.dayofweek >= 5).astype(int),
        }
    )

    product_counts = (
        df.groupby(["id_bon", "retail_product_name"]).size().unstack(fill_value=0)
    )
    if binary_product_counts:
        product_counts = (product_counts > 0).astype(int)

    sauce_counts = (
        df[df["retail_product_name"].isin(SAUCE_NAMES)]
        .groupby(["id_bon", "retail_product_name"])
        .size()
        .unstack(fill_value=0)
    )
    sauce_labels = sauce_counts.reindex(index=product_counts.index, columns=SAUCE_NAMES, fill_value=0)
    sauce_labels = (sauce_labels > 0).astype(int)

    if drop_sauce_features:
        product_counts = product_counts.drop(columns=SAUCE_NAMES, errors="ignore")

    X = (
        agg_features.join(time_features)
        .join(product_counts)
        .fillna(0)
        .sort_index()
    )

    zero_var_cols = list(X.columns[X.nunique() <= 1])
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)

    meta = {
        "num_receipts": len(X),
        "feature_count": X.shape[1],
        "positive_rates": {s: float(sauce_labels[s].mean()) for s in SAUCE_NAMES},
        "date_range": (receipt_time.min(), receipt_time.max()),
        "dropped_zero_variance": zero_var_cols,
    }

    ordered_index = X.index.sort_values()
    return X.loc[ordered_index], sauce_labels.loc[ordered_index], receipt_time.loc[ordered_index], meta


def build_product_matrix(
    csv_path: Path,
    binary_product_counts: bool = True,
    drop_zero_variance: bool = True,
):
    """
    Build a product count/indicator matrix for all receipts plus price statistics.
    Used for upsell ranking where we model P(product | partial cart).
    """
    df = pd.read_csv(csv_path)
    df["data_bon"] = pd.to_datetime(df["data_bon"])

    receipt_time = df.groupby("id_bon")["data_bon"].min()

    product_counts = (
        df.groupby(["id_bon", "retail_product_name"]).size().unstack(fill_value=0)
    )
    if binary_product_counts:
        product_counts = (product_counts > 0).astype(int)

    price_map = (
        df.groupby("retail_product_name")["SalePriceWithVAT"].mean().to_dict()
    )

    if drop_zero_variance:
        zero_var_cols = list(product_counts.columns[product_counts.nunique() <= 1])
        product_counts = product_counts.drop(columns=zero_var_cols)
    else:
        zero_var_cols = []

    meta = {
        "num_receipts": len(product_counts),
        "feature_count": product_counts.shape[1],
        "date_range": (receipt_time.min(), receipt_time.max()),
        "dropped_zero_variance": zero_var_cols,
    }

    ordered_index = product_counts.index.sort_values()
    return product_counts.loc[ordered_index], receipt_time.loc[ordered_index], price_map, meta
