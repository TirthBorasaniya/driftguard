"""Tests for the bounded Great Expectations style validation suite."""

import numpy as np
import pandas as pd
import pytest

from src.validation.expectations import (
    MAX_NULL_RATE,
    MIN_ROW_COUNT,
    NUMERICAL_FEATURE_COLS,
    SUITE_NAME,
    validate_batch,
)


@pytest.fixture
def valid_batch():
    """A clean batch above MIN_ROW_COUNT with no nulls."""
    n = MIN_ROW_COUNT + 500
    np.random.seed(0)
    return pd.DataFrame({col: np.random.rand(n) for col in NUMERICAL_FEATURE_COLS})


def test_suite_name_is_network_flow():
    assert SUITE_NAME == "network_flow_feature_suite"


def test_numerical_feature_cols_count():
    assert len(NUMERICAL_FEATURE_COLS) == 10


def test_valid_batch_passes(valid_batch):
    o_passed, failures = validate_batch(valid_batch)
    assert o_passed, f"expected pass but got: {failures}"


def test_null_rate_above_two_percent_fails(valid_batch):
    df = valid_batch.copy()
    n = len(df)
    # set 5% of one feature column to null (above MAX_NULL_RATE of 2%)
    df.loc[df.index[: int(n * 0.05)], "flow_duration"] = np.nan
    o_passed, failures = validate_batch(df)
    assert not o_passed
    assert any("flow_duration" in f for f in failures)


def test_null_rate_below_two_percent_passes(valid_batch):
    df = valid_batch.copy()
    n = len(df)
    # 1% nulls is within the 2% allowance
    df.loc[df.index[: int(n * 0.01)], "syn_flag_count"] = np.nan
    o_passed, _ = validate_batch(df)
    assert o_passed


def test_row_count_below_minimum_fails():
    df = pd.DataFrame({col: [0.0] * 100 for col in NUMERICAL_FEATURE_COLS})
    o_passed, failures = validate_batch(df)
    assert not o_passed
    assert any("row_count" in f for f in failures)


def test_max_null_rate_constant():
    assert MAX_NULL_RATE == pytest.approx(0.02)
