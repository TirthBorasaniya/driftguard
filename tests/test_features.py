"""Tests for network flow feature engineering: single-event and batch paths."""

import pandas as pd
import pytest

from src.features.engineering import (
    EPSILON,
    FEATURE_COLS,
    compute_features,
    compute_features_batch,
)

# ============= compute_features (serving path) =============


def test_compute_features_returns_exactly_feature_cols(sample_flow_event):
    feature_dict = compute_features(sample_flow_event)
    assert list(feature_dict.keys()) == FEATURE_COLS


def test_compute_features_count_is_ten(sample_flow_event):
    assert len(compute_features(sample_flow_event)) == 10


def test_fwd_bwd_ratio_is_one_for_equal_counts():
    event = {"total_fwd_packets": 7.0, "total_bwd_packets": 7.0}
    assert compute_features(event)["fwd_bwd_packet_ratio"] == pytest.approx(1.0)


def test_fwd_bwd_ratio_uses_epsilon_guard():
    # zero backward packets must not raise; ratio is fwd / EPSILON
    event = {"total_fwd_packets": 5.0, "total_bwd_packets": 0.0}
    assert compute_features(event)["fwd_bwd_packet_ratio"] == pytest.approx(5.0 / EPSILON)


def test_compute_features_passthrough_values(sample_flow_event):
    feature_dict = compute_features(sample_flow_event)
    assert feature_dict["flow_duration"] == pytest.approx(100000.0)
    assert feature_dict["syn_flag_count"] == pytest.approx(1.0)


def test_compute_features_missing_field_defaults_zero():
    feature_dict = compute_features({"total_fwd_packets": 3.0, "total_bwd_packets": 3.0})
    assert feature_dict["flow_duration"] == pytest.approx(0.0)


def test_compute_features_does_not_mutate_input(sample_flow_event):
    original = dict(sample_flow_event)
    compute_features(sample_flow_event)
    assert sample_flow_event == original


# ============= compute_features_batch (training path) =============


def test_compute_features_batch_columns(sample_flow_event):
    df = pd.DataFrame([sample_flow_event, sample_flow_event])
    feature_df = compute_features_batch(df)
    assert list(feature_df.columns) == FEATURE_COLS


def test_compute_features_batch_preserves_index(sample_flow_event):
    df = pd.DataFrame([sample_flow_event, sample_flow_event], index=[10, 20])
    feature_df = compute_features_batch(df)
    assert list(feature_df.index) == [10, 20]


def test_compute_features_batch_ratio_matches_single(sample_flow_event):
    df = pd.DataFrame([sample_flow_event])
    batch_ratio = compute_features_batch(df)["fwd_bwd_packet_ratio"].iloc[0]
    single_ratio = compute_features(sample_flow_event)["fwd_bwd_packet_ratio"]
    assert batch_ratio == pytest.approx(single_ratio)
