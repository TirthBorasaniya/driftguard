"""Data preprocessing pipeline: join, engineer, encode, and temporal split for IEEE-CIS fraud detection."""

import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    BATCH_DIR,
    CATEGORICAL_COLS,
    D_COLS_TO_NORMALIZE,
    ENCODERS_DIR,
    FEATURE_COLS,
    ID_COL,
    NUMERIC_COLS,
    PROCESSED_DIR,
    RAW_IDENTITY_FILE,
    RAW_TRANSACTION_FILE,
    REFERENCE_DIR,
    SPLIT_RATIOS,
    TARGET_COL,
    TIME_COL,
)


# ============= Data Loading =============


def load_raw_data():
    """
    Load raw transaction and identity CSVs and perform left join.

    Returns
    -------
    df : pd.DataFrame
        Joined dataframe with all transaction and identity columns.
    """
    if not RAW_TRANSACTION_FILE.exists():
        raise FileNotFoundError(
            f"Transaction file not found: {RAW_TRANSACTION_FILE}\n"
            "Download from: https://www.kaggle.com/competitions/ieee-fraud-detection/data"
        )

    print(f"Loading transactions from {RAW_TRANSACTION_FILE}")
    df_txn = pd.read_csv(RAW_TRANSACTION_FILE)
    print(f"  Transactions shape: {df_txn.shape}")

    if RAW_IDENTITY_FILE.exists():
        print(f"Loading identity from {RAW_IDENTITY_FILE}")
        df_id = pd.read_csv(RAW_IDENTITY_FILE)
        print(f"  Identity shape: {df_id.shape}")
        df = df_txn.merge(df_id, on=ID_COL, how="left")
        print(f"  Identity match rate: {len(df_id) / len(df_txn):.1%}")
    else:
        print("  Identity file not found, proceeding with transactions only")
        df = df_txn

    print(f"  Joined shape: {df.shape}")
    return df


# ============= Feature Engineering =============


def engineer_features(df):
    """
    Apply numeric transforms and create derived features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw joined dataframe.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with engineered features added.
    """
    print("Engineering features...")

    # log transform of transaction amount to reduce right skew
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    # temporal cycle features from TransactionDT (seconds since reference)
    df["transaction_day"] = (df[TIME_COL] / 86400) % 7
    df["transaction_hour"] = (df[TIME_COL] / 3600) % 24

    # normalize D columns relative to transaction day
    for col in D_COLS_TO_NORMALIZE:
        col_norm = f"{col}n"
        if col in df.columns:
            df[col_norm] = df[col] - df["transaction_day"]
        else:
            df[col_norm] = np.nan

    print(f"  Engineered features added. Shape: {df.shape}")
    return df


# ============= Categorical Encoding =============


def encode_categoricals(df, o_fit_encoders=True):
    """
    Encode categorical columns with LabelEncoder.

    Missing values are filled with 'unknown' before encoding. Unseen values
    at inference time are also mapped to 'unknown'.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with raw categorical values.
    o_fit_encoders : bool
        If True, fit new encoders and save to disk. If False, load existing.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with categorical columns encoded as integers.
    """
    ENCODERS_DIR.mkdir(parents=True, exist_ok=True)

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            df[col] = "unknown"

        df[col] = df[col].fillna("unknown").astype(str)

        encoder_path = ENCODERS_DIR / f"{col}_encoder.pkl"

        if o_fit_encoders:
            le = LabelEncoder()
            unique_vals = sorted(df[col].unique().tolist())
            if "unknown" not in unique_vals:
                unique_vals.append("unknown")
            le.fit(unique_vals)

            with open(encoder_path, "wb") as f:
                pickle.dump(le, f)
            print(f"  Saved encoder: {encoder_path.name} ({len(le.classes_)} classes)")
        else:
            with open(encoder_path, "rb") as f:
                le = pickle.load(f)
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x, kc=known_classes: x if x in kc else "unknown")

        df[col] = le.transform(df[col])

    return df


# ============= Missing Value Imputation =============


def impute_missing(df):
    """
    Fill missing numeric values with -999.

    LightGBM bins -999 as a separate category, handling missingness natively.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with potential missing numeric values.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with no missing values in feature columns.
    """
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(-999)
        else:
            df[col] = -999

    return df


# ============= Temporal Split =============


def temporal_split(df):
    """
    Split dataframe into temporal partitions sorted by TransactionDT.

    All splits are strictly temporal to prevent data leakage. Rows are sorted
    by time and partitioned sequentially according to SPLIT_RATIOS.

    Parameters
    ----------
    df : pd.DataFrame
        Full processed dataframe.

    Returns
    -------
    splits : dict
        Mapping of split name to dataframe.
    """
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    n = len(df)

    boundaries = {}
    cumulative = 0
    for name, ratio in SPLIT_RATIOS.items():
        start = cumulative
        cumulative += int(n * ratio)
        end = min(cumulative, n)
        boundaries[name] = (start, end)

    # last split captures remaining rows
    last_key = list(SPLIT_RATIOS.keys())[-1]
    boundaries[last_key] = (boundaries[last_key][0], n)

    splits = {}
    for name, (start, end) in boundaries.items():
        splits[name] = df.iloc[start:end].copy()
        fraud_rate = splits[name][TARGET_COL].mean() if TARGET_COL in splits[name].columns else 0
        print(f"  {name}: {len(splits[name]):,} rows (fraud rate: {fraud_rate:.3%})")

    return splits


# ============= Save Splits =============


def save_splits(splits):
    """
    Save split dataframes to their respective directories.

    Parameters
    ----------
    splits : dict
        Mapping of split name to dataframe.
    """
    save_cols = [c for c in FEATURE_COLS + [TARGET_COL, ID_COL, TIME_COL]]

    for name, df_split in splits.items():
        available_cols = [c for c in save_cols if c in df_split.columns]

        if name == "train":
            path = PROCESSED_DIR / "train.csv"
        elif name == "test":
            path = PROCESSED_DIR / "test.csv"
        elif name == "reference":
            path = REFERENCE_DIR / "reference.csv"
        elif name.startswith("batch_"):
            path = BATCH_DIR / f"{name}.csv"
        else:
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        df_split[available_cols].to_csv(path, index=False)
        print(f"  Saved {name}: {path} ({len(df_split):,} rows)")


# ============= Main Pipeline =============


def run_preprocessing():
    """Execute the full preprocessing pipeline."""
    print("=" * 60)
    print("DriftGuard: IEEE-CIS Fraud Detection Preprocessing")
    print("=" * 60)

    df = load_raw_data()
    df = engineer_features(df)
    df = encode_categoricals(df, o_fit_encoders=True)
    df = impute_missing(df)

    print("\nSplitting data temporally...")
    splits = temporal_split(df)

    print("\nSaving splits...")
    save_splits(splits)

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print(f"  Total rows: {len(df):,}")
    print(f"  Fraud rate: {df[TARGET_COL].mean():.2%}")
    print(f"  Features: {len(FEATURE_COLS)} ({len(NUMERIC_COLS)} numeric, {len(CATEGORICAL_COLS)} categorical)")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing()
