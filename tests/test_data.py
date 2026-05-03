"""Tests for data preprocessing, feature engineering, and encoding."""

import numpy as np
import pandas as pd
import pytest

from src.data.encoders import SafeLabelEncoder
from src.features.engineering import engineer_features, haversine_km


def test_haversine_zero_distance():
    dist = haversine_km(
        np.array([40.0]), np.array([-74.0]),
        np.array([40.0]), np.array([-74.0]),
    )
    assert dist[0] == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance():
    # NYC to London approximately 5570 km
    dist = haversine_km(
        np.array([40.7128]), np.array([-74.0060]),
        np.array([51.5074]), np.array([-0.1278]),
    )
    assert 5400 < dist[0] < 5800


def test_engineer_features_creates_columns(sample_df):
    df = engineer_features(sample_df)
    assert "hour_of_day" in df.columns
    assert "day_of_week" in df.columns
    assert "age" in df.columns
    assert "distance_km" in df.columns
    assert "amt_log" in df.columns


def test_engineer_features_amt_log_nonneg(sample_df):
    df = engineer_features(sample_df)
    assert (df["amt_log"] >= 0).all()


def test_engineer_features_hour_range(sample_df):
    df = engineer_features(sample_df)
    assert df["hour_of_day"].between(0, 23).all()


def test_engineer_features_dow_range(sample_df):
    df = engineer_features(sample_df)
    assert df["day_of_week"].between(0, 6).all()


class TestSafeLabelEncoder:
    def test_fit_transform_known(self):
        enc = SafeLabelEncoder()
        s = pd.Series(["a", "b", "c", "a"])
        enc.fit(s)
        result = enc.transform(s)
        assert set(result.tolist()) == {0, 1, 2}

    def test_unseen_maps_to_negative_one(self):
        enc = SafeLabelEncoder()
        enc.fit(pd.Series(["a", "b"]))
        result = enc.transform(pd.Series(["a", "z", "unknown_val"]))
        assert result.iloc[1] == -1
        assert result.iloc[2] == -1

    def test_null_maps_to_negative_one(self):
        enc = SafeLabelEncoder()
        enc.fit(pd.Series(["a", "b"]))
        result = enc.transform(pd.Series(["a", None, np.nan]))
        assert result.iloc[1] == -1
        assert result.iloc[2] == -1

    def test_save_load(self, tmp_path):
        enc = SafeLabelEncoder()
        enc.fit(pd.Series(["x", "y", "z"]))
        path = tmp_path / "enc.pkl"
        enc.save(path)
        loaded = SafeLabelEncoder.load(path)
        result = loaded.transform(pd.Series(["x", "new"]))
        assert result.iloc[0] >= 0
        assert result.iloc[1] == -1
