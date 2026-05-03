"""Tests for data preprocessing module."""

import numpy as np
import pandas as pd
import pytest

import src.config as cfg
from src.config import CATEGORICAL_COLS, D_COLS_TO_NORMALIZE, NUMERIC_COLS
from src.data.preprocess import (
    encode_categoricals,
    engineer_features,
    impute_missing,
    temporal_split,
)


def test_engineer_features_creates_columns(sample_transaction_df):
    """Feature engineering must create all expected derived columns."""
    df = engineer_features(sample_transaction_df.copy())
    assert "TransactionAmt_log" in df.columns
    assert "transaction_day" in df.columns
    assert "transaction_hour" in df.columns
    for col in D_COLS_TO_NORMALIZE:
        assert f"{col}n" in df.columns


def test_transaction_amt_log_correct(sample_transaction_df):
    """Log-transformed amount must equal log1p of raw amount."""
    df = engineer_features(sample_transaction_df.copy())
    assert (df["TransactionAmt_log"] >= 0).all()
    np.testing.assert_allclose(
        df["TransactionAmt_log"],
        np.log1p(sample_transaction_df["TransactionAmt"]),
    )


def test_temporal_features_range(sample_transaction_df):
    """Temporal cycle features must be within expected ranges."""
    df = engineer_features(sample_transaction_df.copy())
    assert df["transaction_day"].between(0, 7).all()
    assert df["transaction_hour"].between(0, 24).all()


def test_d_normalization(sample_transaction_df):
    """Normalized D columns must equal D - transaction_day."""
    df = engineer_features(sample_transaction_df.copy())
    for col in D_COLS_TO_NORMALIZE:
        if col in sample_transaction_df.columns:
            col_norm = f"{col}n"
            mask = df[col].notna()
            if mask.any():
                expected = df.loc[mask, col] - df.loc[mask, "transaction_day"]
                np.testing.assert_allclose(df.loc[mask, col_norm], expected)


def test_encode_categoricals_produces_integers(sample_transaction_df, tmp_path):
    """Encoded categorical columns must be integer type."""
    original_dir = cfg.ENCODERS_DIR
    cfg.ENCODERS_DIR = tmp_path

    try:
        df = engineer_features(sample_transaction_df.copy())
        df = encode_categoricals(df, o_fit_encoders=True)
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                assert df[col].dtype in [np.int64, np.int32, int, np.intp]
    finally:
        cfg.ENCODERS_DIR = original_dir


def test_encoder_round_trip(sample_transaction_df, tmp_path):
    """Encoding then loading encoders must produce consistent results."""
    original_dir = cfg.ENCODERS_DIR
    cfg.ENCODERS_DIR = tmp_path

    try:
        df1 = engineer_features(sample_transaction_df.copy())
        df1 = encode_categoricals(df1, o_fit_encoders=True)

        df2 = engineer_features(sample_transaction_df.copy())
        df2 = encode_categoricals(df2, o_fit_encoders=False)

        for col in CATEGORICAL_COLS:
            if col in df1.columns and col in df2.columns:
                pd.testing.assert_series_equal(df1[col], df2[col])
    finally:
        cfg.ENCODERS_DIR = original_dir


def test_impute_missing_no_nans(sample_transaction_df):
    """After imputation, no numeric feature column should contain NaN."""
    df = engineer_features(sample_transaction_df.copy())
    df = impute_missing(df)
    for col in NUMERIC_COLS:
        if col in df.columns:
            assert not df[col].isna().any(), f"NaN found in {col}"


def test_impute_missing_fills_with_negative_999(sample_transaction_df):
    """Missing numeric values must be filled with -999."""
    df = engineer_features(sample_transaction_df.copy())
    # introduce a known NaN
    df.loc[0, "dist1"] = np.nan
    df = impute_missing(df)
    assert df.loc[0, "dist1"] == -999


def test_temporal_split_preserves_order(sample_transaction_df):
    """Temporal splits must be non-overlapping and time-ordered."""
    df = engineer_features(sample_transaction_df.copy())
    splits = temporal_split(df)

    max_times = []
    for name in ["train", "test", "reference", "batch_0", "batch_1", "batch_2"]:
        if name in splits and len(splits[name]) > 0:
            max_times.append(splits[name]["TransactionDT"].max())

    # each subsequent split's max time should be >= previous
    for i in range(1, len(max_times)):
        assert max_times[i] >= max_times[i - 1]


def test_temporal_split_covers_all_rows(sample_transaction_df):
    """All rows must be assigned to exactly one split."""
    df = engineer_features(sample_transaction_df.copy())
    splits = temporal_split(df)
    total = sum(len(s) for s in splits.values())
    assert total == len(df)


@pytest.mark.requires_data
def test_raw_data_schema():
    """Raw CSV files must contain expected key columns."""
    df_txn = pd.read_csv(cfg.RAW_TRANSACTION_FILE, nrows=5)
    df_id = pd.read_csv(cfg.RAW_IDENTITY_FILE, nrows=5)
    assert "TransactionID" in df_txn.columns
    assert "TransactionID" in df_id.columns
    assert "isFraud" in df_txn.columns
    assert "TransactionDT" in df_txn.columns
