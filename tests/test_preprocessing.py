"""Tests for data preprocessing: imputation, temporal split, and the Feast source builder."""

import numpy as np
import pandas as pd
import pytest

from src.config import ENTITY_COL, FEATURE_COLS
from src.data.preprocess import (
    INVALID_FLOW_BOUNDED_COLS,
    build_src_ip_stats,
    filter_invalid_flow_rows,
    impute_numeric,
    temporal_split,
)


def test_impute_numeric_fills_missing(sample_feature_df):
    df = sample_feature_df.copy()
    df.loc[df.index[0], "flow_duration"] = np.nan
    result = impute_numeric(df)
    for col in FEATURE_COLS:
        assert not result[col].isna().any(), f"NaN remains in {col}"


def test_impute_numeric_fill_value_is_zero(sample_feature_df):
    df = sample_feature_df.copy()
    df.loc[df.index[0], "flow_duration"] = np.nan
    result = impute_numeric(df)
    assert result.loc[result.index[0], "flow_duration"] == pytest.approx(0.0)


def test_temporal_split_returns_three_keys(sample_feature_df):
    splits = temporal_split(sample_feature_df)
    assert set(splits.keys()) == {"train", "test", "stream"}


def test_temporal_split_covers_all_rows(sample_feature_df):
    splits = temporal_split(sample_feature_df)
    total = sum(len(s) for s in splits.values())
    assert total == len(sample_feature_df)


def test_temporal_split_no_overlap(sample_feature_df):
    splits = temporal_split(sample_feature_df)
    train_max = pd.to_datetime(splits["train"]["event_timestamp"]).max()
    test_min = pd.to_datetime(splits["test"]["event_timestamp"]).min()
    assert train_max <= test_min


def test_temporal_split_train_fraction(sample_feature_df):
    splits = temporal_split(sample_feature_df)
    frac = len(splits["train"]) / len(sample_feature_df)
    assert 0.70 <= frac <= 0.80


def test_build_src_ip_stats_one_row_per_entity(sample_feature_df):
    stats = build_src_ip_stats(sample_feature_df)
    assert stats[ENTITY_COL].nunique() == len(stats)


def test_build_src_ip_stats_columns(sample_feature_df):
    stats = build_src_ip_stats(sample_feature_df)
    for col in FEATURE_COLS:
        assert col in stats.columns
    assert "event_timestamp" in stats.columns
    assert "created_timestamp" in stats.columns


def test_filter_invalid_flow_rows_drops_negative_values(sample_feature_df, capsys):
    df = sample_feature_df.copy()
    df.loc[df.index[0], "flow_duration"] = -1.0
    result = filter_invalid_flow_rows(df, INVALID_FLOW_BOUNDED_COLS)

    assert len(result) == len(df) - 1
    assert (result["flow_duration"] >= 0).all()
    assert "Dropped 1 row(s)" in capsys.readouterr().out


def test_filter_invalid_flow_rows_retains_valid_rows(sample_feature_df):
    df = sample_feature_df.copy()
    result = filter_invalid_flow_rows(df, INVALID_FLOW_BOUNDED_COLS)
    assert len(result) == len(df)
