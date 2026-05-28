"""Tests for feature engineering: batch transforms and single-event serving path."""

import math

import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_COLS, FEATURE_COLS, NUMERIC_COLS
from src.features.engineering import (
    compute_age,
    engineer_features,
    engineer_single_event,
    haversine_km,
)


# ============= haversine_km =============


def test_haversine_same_point_is_zero():
    dist = haversine_km(
        np.array([40.0]), np.array([-74.0]),
        np.array([40.0]), np.array([-74.0]),
    )
    assert dist[0] == pytest.approx(0.0, abs=1e-6)


def test_haversine_nyc_to_london():
    dist = haversine_km(
        np.array([40.7128]), np.array([-74.0060]),
        np.array([51.5074]), np.array([-0.1278]),
    )
    assert 5400 < dist[0] < 5800


def test_haversine_returns_nonnegative():
    lats = np.random.uniform(-90, 90, 50)
    lons = np.random.uniform(-180, 180, 50)
    dist = haversine_km(lats, lons, lats[::-1], lons[::-1])
    assert (dist >= 0).all()


# ============= compute_age =============


def test_compute_age_known_value():
    dob = pd.Series(["1990-01-01"])
    ts = pd.Series(["2020-01-01"])
    age = compute_age(dob, ts)
    assert age.iloc[0] == pytest.approx(30.0, abs=0.1)


def test_compute_age_invalid_fills_sentinel():
    dob = pd.Series(["not-a-date"])
    ts = pd.Series(["2020-01-01"])
    age = compute_age(dob, ts)
    assert age.iloc[0] == pytest.approx(-999.0)


# ============= engineer_features (batch) =============


def test_engineer_features_adds_derived_columns(sample_df):
    df = engineer_features(sample_df)
    for col in ("hour_of_day", "day_of_week", "age", "distance_km", "amt_log"):
        assert col in df.columns, f"Missing column: {col}"


def test_engineer_features_hour_in_range(sample_df):
    df = engineer_features(sample_df)
    assert df["hour_of_day"].between(0, 23).all()


def test_engineer_features_dow_in_range(sample_df):
    df = engineer_features(sample_df)
    assert df["day_of_week"].between(0, 6).all()


def test_engineer_features_amt_log_nonneg(sample_df):
    df = engineer_features(sample_df)
    assert (df["amt_log"] >= 0).all()


def test_engineer_features_does_not_mutate_input(sample_df):
    original_cols = set(sample_df.columns)
    engineer_features(sample_df)
    assert set(sample_df.columns) == original_cols


# ============= engineer_single_event (serving path) =============


@pytest.fixture
def valid_event():
    return {
        "trans_date_trans_time": "2020-06-21 12:14:25",
        "cc_num": "4532015112830366",
        "merchant": "merchant_A",
        "category": "grocery_pos",
        "amt": 149.62,
        "gender": "F",
        "city": "Henderson",
        "state": "TX",
        "zip": "76054",
        "lat": 36.0788,
        "long": -81.1781,
        "city_pop": 35550,
        "job": "Engineer",
        "dob": "1987-01-01",
        "merch_lat": 36.011293,
        "merch_long": -82.048315,
    }


def test_single_event_produces_derived_fields(valid_event):
    result = engineer_single_event(valid_event)
    for field in ("hour_of_day", "day_of_week", "age", "distance_km", "amt_log"):
        assert field in result, f"Missing field: {field}"


def test_single_event_hour_at_noon(valid_event):
    result = engineer_single_event(valid_event)
    assert result["hour_of_day"] == pytest.approx(12.0)


def test_single_event_amt_log_is_log1p(valid_event):
    result = engineer_single_event(valid_event)
    expected = math.log1p(149.62)
    assert result["amt_log"] == pytest.approx(expected, rel=1e-5)


def test_single_event_missing_timestamp_uses_sentinel():
    result = engineer_single_event({"amt": 50.0})
    assert result["hour_of_day"] == -999.0
    assert result["day_of_week"] == -999.0


def test_single_event_missing_coords_uses_sentinel():
    result = engineer_single_event({"amt": 50.0})
    assert result["distance_km"] == -999.0


def test_single_event_does_not_mutate_input(valid_event):
    original = dict(valid_event)
    engineer_single_event(valid_event)
    assert valid_event == original


# ============= Feature column definitions =============


def test_feature_cols_include_all_numeric():
    for col in NUMERIC_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"


def test_feature_cols_include_all_categorical():
    for col in CATEGORICAL_COLS:
        assert col in FEATURE_COLS, f"{col} missing from FEATURE_COLS"


def test_no_duplicate_feature_cols():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_no_overlap_between_numeric_and_categorical():
    overlap = set(NUMERIC_COLS) & set(CATEGORICAL_COLS)
    assert len(overlap) == 0, f"Overlap: {overlap}"
