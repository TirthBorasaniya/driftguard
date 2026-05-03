"""Data preprocessing: load Sparkov CSV, engineer features, encode categoricals, temporal split."""

import json

import numpy as np
import pandas as pd

from src.config import (
    CARD_STATS_FILE,
    CATEGORICAL_COLS,
    CATEGORY_STATS_FILE,
    ENCODERS_DIR,
    FEATURE_COLS,
    NUMERIC_COLS,
    PROCESSED_DIR,
    REFERENCE_FILE,
    STREAM_FILE,
    TARGET_COL,
    TEST_FILE,
    TIME_COL,
    TRAIN_CSV,
    TRAIN_FILE,
)
from src.data.encoders import fit_and_save_encoders
from src.features.engineering import engineer_features


# ============= Loading =============


def load_raw(path=None) -> pd.DataFrame:
    """
    Load the Sparkov fraudTrain CSV.

    Parameters
    ----------
    path : Path or None
        Override path. Defaults to TRAIN_CSV.

    Returns
    -------
    df : pd.DataFrame
        Raw dataframe with all Sparkov columns.
    """
    path = path or TRAIN_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}\n"
            "Download from: https://www.kaggle.com/datasets/kartik2112/fraud-detection"
        )
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Fraud rate: {df[TARGET_COL].mean():.4%}")
    return df


# ============= Temporal Split =============


def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split dataframe into temporal partitions at fixed percentile boundaries.

    Boundaries
    ----------
    train     : rows 0 to P75 (75%)
    test      : rows P75 to P90 (15%)
    reference : rows P90 to P95 (5%)  -- Evidently baseline
    stream    : rows P95 to end (5%)  -- Kafka replay

    Parameters
    ----------
    df : pd.DataFrame
        Full preprocessed dataframe, sorted by trans_date_trans_time.

    Returns
    -------
    splits : dict
        Mapping of split name to dataframe.
    """
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)

    p75 = int(n * 0.75)
    p90 = int(n * 0.90)
    p95 = int(n * 0.95)

    splits = {
        "train": df.iloc[:p75].copy(),
        "test": df.iloc[p75:p90].copy(),
        "reference": df.iloc[p90:p95].copy(),
        "stream": df.iloc[p95:].copy(),
    }

    for name, split in splits.items():
        fraud_rate = split[TARGET_COL].mean() if TARGET_COL in split.columns else 0.0
        print(f"  {name}: {len(split):,} rows (fraud: {fraud_rate:.4%})")

    return splits


# ============= Aggregated Feature Tables for Feast =============


def build_card_stats(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling 7-day transaction statistics per cc_num from training data.

    These serve as the offline feature store for Feast card_stats_7d view.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training split.

    Returns
    -------
    stats : pd.DataFrame
        Aggregated stats with event_timestamp column.
    """
    df = df_train.copy()
    df["trans_ts"] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(["cc_num", "trans_ts"])

    agg = df.groupby("cc_num").agg(
        txn_count_7d=("amt", "count"),
        amt_mean_7d=("amt", "mean"),
        amt_max_7d=("amt", "max"),
    ).reset_index()

    agg["txn_velocity_7d"] = (agg["txn_count_7d"] / 7.0).astype("float32")
    agg["event_timestamp"] = df.groupby("cc_num")["trans_ts"].max().values
    agg["created_timestamp"] = pd.Timestamp.now()

    return agg


def build_category_stats(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fraud statistics per merchant category from training data.

    These serve as the offline feature store for Feast category_fraud_rate view.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training split.

    Returns
    -------
    stats : pd.DataFrame
        Per-category fraud stats with event_timestamp column.
    """
    agg = df_train.groupby("category").agg(
        category_fraud_count=(TARGET_COL, "sum"),
        category_txn_count=(TARGET_COL, "count"),
    ).reset_index()

    agg["category_fraud_rate"] = (
        agg["category_fraud_count"] / agg["category_txn_count"].clip(lower=1)
    ).astype("float32")

    agg["event_timestamp"] = pd.Timestamp.now()
    agg["created_timestamp"] = pd.Timestamp.now()

    return agg


# ============= Imputation =============


def impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing numeric values with -999 (LightGBM handles as separate bin)."""
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(-999).astype("float32")
        else:
            df[col] = np.float32(-999)
    return df


# ============= Main Pipeline =============


def run_preprocessing() -> None:
    """Execute full preprocessing pipeline and save all splits to data/processed/."""
    print("=" * 60)
    print("Fraud Pipeline: Preprocessing")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df = engineer_features(df)

    print("\nFitting encoders...")
    encoders_dict = fit_and_save_encoders(df, CATEGORICAL_COLS, ENCODERS_DIR)

    for col, enc in encoders_dict.items():
        df[col] = enc.transform(df[col])

    df = impute_numeric(df)

    print("\nSplitting temporally...")
    splits = temporal_split(df)

    print("\nSaving splits...")
    splits["train"].to_parquet(TRAIN_FILE, index=False)
    splits["test"].to_parquet(TEST_FILE, index=False)
    splits["reference"].to_parquet(REFERENCE_FILE, index=False)
    splits["stream"].to_parquet(STREAM_FILE, index=False)
    print(f"  Saved: train, test, reference, stream -> {PROCESSED_DIR}")

    print("\nBuilding Feast aggregation tables...")
    card_stats = build_card_stats(splits["train"])
    card_stats.to_parquet(CARD_STATS_FILE, index=False)
    print(f"  Saved card_stats_7d: {len(card_stats):,} rows")

    cat_stats = build_category_stats(splits["train"])
    cat_stats.to_parquet(CATEGORY_STATS_FILE, index=False)
    print(f"  Saved category_fraud_rate: {len(cat_stats):,} rows")

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print(f"  Total rows: {len(df):,}")
    print(f"  Features: {len(FEATURE_COLS)}")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing()
