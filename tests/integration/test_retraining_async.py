"""Integration: a retraining run does not slow the consumer by more than 10% (SPEC 10)."""

import threading
import time

import pytest

from driftguard.common.settings import get_settings
from driftguard.orchestrator.main import flow_run_state, submit_deployment

from .helpers import DATASET, metric_value, produce_synthetic

pytestmark = pytest.mark.integration
METRICS = "http://localhost:9101/metrics"
RATE_EPS = 500
MEASURE_S = 60


def consumption_rate(duration_s: float) -> float:
    """Events consumed per second over a measurement interval."""
    start = metric_value(METRICS, "driftguard_events_consumed_total", dataset=DATASET)
    time.sleep(duration_s)
    end = metric_value(METRICS, "driftguard_events_consumed_total", dataset=DATASET)
    return (end - start) / duration_s


def test_consumer_rate_within_10pct_during_retraining():
    settings = get_settings()
    n_events = RATE_EPS * (3 * MEASURE_S + 60)
    thread = threading.Thread(target=produce_synthetic, args=(n_events, 0.0, RATE_EPS))
    thread.start()
    time.sleep(20)
    baseline = consumption_rate(MEASURE_S)
    run_id = submit_deployment(
        "retrain/retrain-on-drift",
        {
            "dataset": DATASET,
            "trigger_id": f"it-{int(time.time())}",
            "window_id": "it",
            "window_end_ms": int(time.time() * 1000),
        },
    )
    time.sleep(5)
    during = consumption_rate(MEASURE_S)
    state = flow_run_state(run_id)
    thread.join()
    assert state is not None
    assert during >= 0.9 * baseline, f"baseline {baseline:.1f} eps, during {during:.1f} eps"
    assert settings.dataset == DATASET
