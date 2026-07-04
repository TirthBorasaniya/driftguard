"""Tests for point-in-time correct Feast offline materialization (infra improvement 8)."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

pytest.importorskip("feast")

from src.config import FEATURE_COLS, FEATURE_REPO_DIR  # noqa: E402
from src.features.materializer import (  # noqa: E402
    MATERIALIZATION_LOOKBACK_HOURS,
    materialize_offline_features,
)


@pytest.fixture
def populated_offline_store(tmp_path, monkeypatch):
    n = 20
    now = datetime.now(timezone.utc)
    df = pd.DataFrame(
        {
            "src_ip": [f"10.0.0.{i % 5}" for i in range(n)],
            "event_timestamp": [now - timedelta(hours=i % 3) for i in range(n)],
            "created_timestamp": [now] * n,
            **{col: [float(i) for i in range(n)] for col in FEATURE_COLS},
        }
    )
    stats_path = tmp_path / "network_flow_features.parquet"
    df.to_parquet(stats_path)
    monkeypatch.setattr("src.config.NETWORK_FLOW_STATS_FILE", stats_path)
    return stats_path


def test_materialize_offline_features_completes_and_produces_output(populated_offline_store):
    features_df = materialize_offline_features(
        str(FEATURE_REPO_DIR), lookback_hours=MATERIALIZATION_LOOKBACK_HOURS
    )
    assert len(features_df) > 0
    assert "src_ip" in features_df.columns
