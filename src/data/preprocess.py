"""Data preprocessing: load CICIDS2017, compute features, temporal split, build Feast source."""

import pandas as pd

from src.config import (
    CICIDS_COLUMN_MAP,
    CICIDS_DATA_DIR,
    CICIDS_FILES_ORDERED,
    ENTITY_COL,
    FEATURE_COLS,
    NETWORK_FLOW_STATS_FILE,
    PROCESSED_DIR,
    STREAM_FILE,
    TARGET_COL,
    TEST_FILE,
    TRAIN_FILE,
)
from src.features.engineering import compute_features_batch
from src.producer.flow_producer import TIMESTAMP_FORMAT, load_cicids_file

# datetime column used for the point-in-time temporal split and Feast source
EVENT_TS_COL = "event_timestamp"


# ============= Loading =============


def load_raw() -> pd.DataFrame:
    """
    Load and concatenate all available CICIDS2017 capture files.

    Returns
    -------
    df : pd.DataFrame
        Combined raw dataframe with renamed columns and label_binary added.
    """
    frame_list = []
    for file_name in CICIDS_FILES_ORDERED:
        path = CICIDS_DATA_DIR / file_name
        if not path.exists():
            print(f"  Missing (skipped): {path}")
            continue
        df = load_cicids_file(str(path), CICIDS_COLUMN_MAP)
        frame_list.append(df)
        print(f"  Loaded {len(df):,} flows from {file_name}")

    if not frame_list:
        raise FileNotFoundError(
            f"No CICIDS2017 files found under {CICIDS_DATA_DIR}\n"
            "Download from: https://www.unb.ca/cic/datasets/ids-2017.html"
        )

    df = pd.concat(frame_list, ignore_index=True)
    print(f"Loaded total: {len(df):,} rows | attack rate: {df['label_binary'].mean():.4%}")
    return df


# ============= Feature Frame =============


def build_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute model features and attach target, entity, and event timestamp.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Combined raw CICIDS2017 dataframe with renamed columns.

    Returns
    -------
    feature_df : pd.DataFrame
        FEATURE_COLS plus label_binary, src_ip, and event_timestamp columns.
    """
    feature_df = compute_features_batch(raw_df)
    feature_df[TARGET_COL] = raw_df["label_binary"].astype(int).values
    feature_df[ENTITY_COL] = raw_df.get("src_ip", "").astype(str).values

    parsed = pd.to_datetime(raw_df.get("timestamp_raw"), format=TIMESTAMP_FORMAT, errors="coerce")
    if parsed.isna().any():
        parsed = parsed.fillna(pd.to_datetime(raw_df.get("timestamp_raw"), errors="coerce"))
    feature_df[EVENT_TS_COL] = parsed.values

    return feature_df


# ============= Temporal Split =============


def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split the dataframe into temporal partitions at fixed percentile boundaries.

    Boundaries
    ----------
    train  : rows 0 to P75 (75%)
    test   : rows P75 to P90 (15%)
    stream : rows P90 to end (10%)  -- Kafka replay

    Never random-split time-ordered flow data; chronological order is preserved
    to avoid leakage across train, test, and the streamed window.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataframe with an event_timestamp column.

    Returns
    -------
    splits : dict
        Mapping of split name to dataframe.
    """
    df = df.sort_values(EVENT_TS_COL).reset_index(drop=True)
    n = len(df)

    p75 = int(n * 0.75)
    p90 = int(n * 0.90)

    splits = {
        "train": df.iloc[:p75].copy(),
        "test": df.iloc[p75:p90].copy(),
        "stream": df.iloc[p90:].copy(),
    }

    for name, split in splits.items():
        attack_rate = split[TARGET_COL].mean() if TARGET_COL in split.columns else 0.0
        print(f"  {name}: {len(split):,} rows (attack: {attack_rate:.4%})")

    return splits


# ============= Imputation =============


def impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing feature values with 0.0, mirroring compute_features defaults."""
    df = df.copy()
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")
        else:
            df[col] = pd.Series(0.0, index=df.index, dtype="float32")
    return df


# ============= Aggregated Feature Table for Feast =============


def build_src_ip_stats(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean flow feature profile per source IP from training data.

    Backs the Feast network_flow_features offline source, keyed by src_ip.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training split with FEATURE_COLS, src_ip, and event_timestamp columns.

    Returns
    -------
    stats : pd.DataFrame
        One row per src_ip with mean features and an event_timestamp column.
    """
    agg = df_train.groupby(ENTITY_COL)[FEATURE_COLS].mean().reset_index()
    agg[EVENT_TS_COL] = df_train.groupby(ENTITY_COL)[EVENT_TS_COL].max().values
    agg["created_timestamp"] = pd.Timestamp.now()
    return agg


# ============= Main Pipeline =============


def run_preprocessing() -> None:
    """Execute full preprocessing and save all splits to data/processed/."""
    print("=" * 60)
    print("Network Anomaly Pipeline: Preprocessing")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    df = build_feature_frame(raw)
    df = impute_numeric(df)

    print("\nSplitting temporally...")
    splits = temporal_split(df)

    print("\nSaving splits...")
    splits["train"].to_parquet(TRAIN_FILE, index=False)
    splits["test"].to_parquet(TEST_FILE, index=False)
    splits["stream"].to_parquet(STREAM_FILE, index=False)
    print(f"  Saved: train, test, stream -> {PROCESSED_DIR}")

    print("\nBuilding Feast aggregation table...")
    stats = build_src_ip_stats(splits["train"])
    stats.to_parquet(NETWORK_FLOW_STATS_FILE, index=False)
    print(f"  Saved network_flow_features: {len(stats):,} rows")

    print("\n" + "=" * 60)
    print("Preprocessing complete.")
    print(f"  Total rows: {len(df):,}")
    print(f"  Features: {len(FEATURE_COLS)}")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing()
