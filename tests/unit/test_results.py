"""Tests for the results artifact writer."""

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from driftguard.common.results import build_results, load_results, write_results

pytestmark = pytest.mark.unit


def test_build_results_has_convention_fields(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("version: 1\n")
    doc = build_results(
        "e9_demo", {"a": 1}, config, notes="n", now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    )
    for key in [
        "schema_version",
        "run_id",
        "git_sha",
        "timestamp_utc",
        "config_path",
        "config_sha256",
        "environment",
        "metrics",
        "notes",
    ]:
        assert key in doc
    assert doc["run_id"] == "2026-01-02T03-04-05Z_e9_demo"
    assert doc["timestamp_utc"] == "2026-01-02T03:04:05Z"
    assert len(doc["config_sha256"]) == 64
    assert "python" in doc["environment"] and "hardware" in doc["environment"]


def test_write_and_load_roundtrip_with_numpy(tmp_path):
    path = write_results(
        "x_test", {"v": np.float64(0.5), "arr": np.arange(3)}, None, results_dir=tmp_path
    )
    data = json.loads(path.read_text())
    assert data["metrics"] == {"v": 0.5, "arr": [0, 1, 2]}
    assert load_results("x_test", tmp_path)["metrics"]["v"] == 0.5
    assert load_results("missing", tmp_path) is None
