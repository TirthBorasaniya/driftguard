"""Tests for the Feast network flow feature view (skipped when feast is unavailable)."""

from datetime import timedelta

import pytest

pytest.importorskip("feast")

from src.features.engineering import FEATURE_COLS
from src.features.feature_repo.feature_views import network_flow_features


def test_feature_view_contains_all_ten_feature_columns():
    field_names = [f.name for f in network_flow_features.schema]
    assert set(field_names) == set(FEATURE_COLS)
    assert len(field_names) == 10


def test_feature_view_ttl_is_one_hour():
    assert network_flow_features.ttl == timedelta(hours=1)


def test_feature_view_name():
    assert network_flow_features.name == "network_flow_features"
