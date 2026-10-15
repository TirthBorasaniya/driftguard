"""E2E: rollback restores the previous champion within 15 seconds (SPEC 10)."""

import subprocess
import time

import pytest

from driftguard.common.mlflow_client import CHAMPION, PREVIOUS, MlflowRegistry, model_name
from driftguard.common.settings import REPO_ROOT, get_settings
from tests.integration.helpers import DATASET, metric_value, wait_for

pytestmark = pytest.mark.e2e
RESTORE_BUDGET_S = 15


def test_rollback_restores_previous_champion():
    settings = get_settings()
    registry = MlflowRegistry(settings.mlflow_tracking_uri)
    name = model_name(DATASET)
    original = registry.get_alias(name, CHAMPION)
    assert original is not None
    # create a newer champion so there is something to roll back from
    subprocess.run(
        [
            "docker",
            "compose",
            "run",
            "--rm",
            "bootstrap",
            "python",
            "scripts/bootstrap.py",
            "--source",
            "synthetic",
            "--o-force-champion",
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    newer = registry.get_alias(name, CHAMPION)
    assert newer.version != original.version
    registry.set_alias(name, PREVIOUS, original.version)
    serving_metrics = f"{settings.serving_url}/metrics"
    assert wait_for(
        lambda: metric_value(
            serving_metrics, "driftguard_model_info", alias="champion", version=newer.version
        )
        == 1.0,
        timeout_s=30,
    )

    started = time.monotonic()
    subprocess.run(["make", "rollback", f"DATASET={DATASET}"], check=True, cwd=REPO_ROOT)
    restored = wait_for(
        lambda: metric_value(
            serving_metrics, "driftguard_model_info", alias="champion", version=original.version
        )
        == 1.0,
        timeout_s=RESTORE_BUDGET_S,
        interval_s=0.5,
    )
    elapsed = time.monotonic() - started
    assert restored and elapsed <= RESTORE_BUDGET_S, f"restored={restored} after {elapsed:.1f}s"
    assert registry.get_alias(name, CHAMPION).version == original.version
