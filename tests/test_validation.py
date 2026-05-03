"""Tests for Great Expectations data validation suite."""

import numpy as np
import pandas as pd
import pytest

from src.validation.expectations import run_validation_suite


def test_valid_data_passes(sample_df):
    passed, failures = run_validation_suite(sample_df)
    assert passed, f"Expected pass but got failures: {failures}"


def test_null_amt_fails(sample_df):
    df = sample_df.copy()
    df.loc[0, "amt"] = None
    passed, failures = run_validation_suite(df)
    assert not passed
    assert any("amt" in f for f in failures)


def test_negative_amt_fails(sample_df):
    df = sample_df.copy()
    df.loc[0, "amt"] = -5.0
    passed, failures = run_validation_suite(df)
    assert not passed


def test_low_category_cardinality_fails(sample_df):
    df = sample_df.copy()
    df["category"] = "grocery_pos"  # only 1 unique value
    passed, failures = run_validation_suite(df)
    assert not passed
    assert any("category" in f for f in failures)


def test_fraud_rate_too_high_fails(sample_df):
    df = sample_df.copy()
    df["is_fraud"] = 1  # 100% fraud rate
    passed, failures = run_validation_suite(df)
    assert not passed
    assert any("is_fraud" in f for f in failures)


def test_duplicate_trans_num_fails(sample_df):
    df = sample_df.copy()
    df["trans_num"] = "dup_txn_000"
    passed, failures = run_validation_suite(df)
    assert not passed
    assert any("trans_num" in f for f in failures)
