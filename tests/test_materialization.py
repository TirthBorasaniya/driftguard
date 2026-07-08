"""Tests for point-in-time correct Feast offline materialization (infra improvement 8)."""

import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

pytest.importorskip("feast")

from src.config import (  # noqa: E402
    FEATURE_COLS,
    FEATURE_REPO_DIR,
    NETWORK_FLOW_STATS_FILE,
)
from src.features.materializer import (  # noqa: E402
    MATERIALIZATION_LOOKBACK_HOURS,
    _feast_executable,
    materialize_offline_features,
)


@pytest.fixture
def populated_offline_store():
    """
    Write synthetic data to the real configured offline store path and run a
    real `feast apply` so the feature view is registered, since Feast's
    FileSource is bound to NETWORK_FLOW_STATS_FILE at import time and cannot
    be monkeypatched after the feature view object is constructed. Backs up
    and restores any pre-existing file/registry at teardown.
    """
    backup_path = NETWORK_FLOW_STATS_FILE.with_suffix(".bak")
    had_existing = NETWORK_FLOW_STATS_FILE.exists()
    if had_existing:
        shutil.copy(NETWORK_FLOW_STATS_FILE, backup_path)

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
    NETWORK_FLOW_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(NETWORK_FLOW_STATS_FILE)

    result = subprocess.run(
        [_feast_executable(), "apply"],
        cwd=str(FEATURE_REPO_DIR),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"feast apply failed:\n{result.stderr}"

    yield NETWORK_FLOW_STATS_FILE

    if had_existing:
        shutil.copy(backup_path, NETWORK_FLOW_STATS_FILE)
        backup_path.unlink()
    else:
        NETWORK_FLOW_STATS_FILE.unlink(missing_ok=True)


def test_materialize_offline_features_completes_and_produces_output(populated_offline_store):
    features_df = materialize_offline_features(
        str(FEATURE_REPO_DIR), lookback_hours=MATERIALIZATION_LOOKBACK_HOURS
    )
    assert len(features_df) > 0
    assert "src_ip" in features_df.columns
