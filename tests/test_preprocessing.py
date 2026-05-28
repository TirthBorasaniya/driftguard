"""Tests for data preprocessing: imputation, temporal split, and aggregation builders."""

import numpy as np
import pandas as pd
import pytest

from src.config import NUMERIC_COLS, TARGET_COL
from src.data.preprocess import build_card_stats, build_category_stats, impute_numeric, temporal_split


@pytest.fixture
def minimal_train_df(sample_df):
    """sample_df with derived features already present for preprocess tests."""
    from src.features.engineering import engineer_features

    return engineer_features(sample_df)


def test_impute_numeric_fills_missing(minimal_train_df):
    """impute_numeric must replace NaN with -999 in every numeric column."""
    df = minimal_train_df.copy()
    if "amt" in df.columns:
        df.loc[0, "amt"] = np.nan
    result = impute_numeric(df)
    for col in NUMERIC_COLS:
        if col in result.columns:
            assert not result[col].isna().any(), f"NaN found in {col} after imputation"


def test_impute_numeric_fills_sentinel(minimal_train_df):
    """impute_numeric must use -999 as the sentinel fill value."""
    df = minimal_train_df.copy()
    if "amt" in df.columns:
        df.loc[0, "amt"] = np.nan
    result = impute_numeric(df)
    if "amt" in result.columns:
        assert result.loc[0, "amt"] == pytest.approx(-999.0)


def test_temporal_split_returns_four_keys(minimal_train_df):
    """temporal_split must return train, test, reference, and stream keys."""
    splits = temporal_split(minimal_train_df)
    assert set(splits.keys()) == {"train", "test", "reference", "stream"}


def test_temporal_split_covers_all_rows(minimal_train_df):
    """temporal_split must assign every row to exactly one partition."""
    splits = temporal_split(minimal_train_df)
    total = sum(len(s) for s in splits.values())
    assert total == len(minimal_train_df)


def test_temporal_split_no_overlap(minimal_train_df):
    """temporal_split must produce non-overlapping partitions."""
    from src.config import TIME_COL

    splits = temporal_split(minimal_train_df)
    train_max = pd.to_datetime(splits["train"][TIME_COL]).max()
    test_min = pd.to_datetime(splits["test"][TIME_COL]).min()
    assert train_max <= test_min


def test_temporal_split_sizes(minimal_train_df):
    """train split must contain approximately 75% of rows."""
    splits = temporal_split(minimal_train_df)
    n = len(minimal_train_df)
    train_frac = len(splits["train"]) / n
    assert 0.70 <= train_frac <= 0.80


def test_build_card_stats_columns(minimal_train_df):
    """build_card_stats must produce txn_count_7d, amt_mean_7d, amt_max_7d, txn_velocity_7d."""
    stats = build_card_stats(minimal_train_df)
    assert "txn_count_7d" in stats.columns
    assert "amt_mean_7d" in stats.columns
    assert "amt_max_7d" in stats.columns
    assert "txn_velocity_7d" in stats.columns
    assert "event_timestamp" in stats.columns


def test_build_card_stats_one_row_per_card(minimal_train_df):
    """build_card_stats must produce exactly one row per unique cc_num."""
    stats = build_card_stats(minimal_train_df)
    assert stats["cc_num"].nunique() == len(stats)


def test_build_category_stats_columns(minimal_train_df):
    """build_category_stats must include fraud rate and count columns."""
    stats = build_category_stats(minimal_train_df)
    assert "category_fraud_rate" in stats.columns
    assert "category_fraud_count" in stats.columns
    assert "category_txn_count" in stats.columns
    assert "event_timestamp" in stats.columns


def test_build_category_stats_rate_bounds(minimal_train_df):
    """category_fraud_rate values must be in [0, 1]."""
    stats = build_category_stats(minimal_train_df)
    assert (stats["category_fraud_rate"] >= 0).all()
    assert (stats["category_fraud_rate"] <= 1).all()
