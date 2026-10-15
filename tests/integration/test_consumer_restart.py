"""Integration: killing the consumer mid-stream loses and duplicates no predictions."""

import pytest

from .helpers import collect_predictions, compose, metric_value, produce_synthetic, wait_for

pytestmark = pytest.mark.integration


def test_kill_and_restart_consumer_mid_batch():
    import threading

    produced: dict = {}
    thread = threading.Thread(
        target=lambda: produced.update(ids=produce_synthetic(10_000, 0.0, rate_eps=500)[0])
    )
    thread.start()
    assert wait_for(
        lambda: metric_value("http://localhost:9101/metrics", "driftguard_events_consumed_total")
        > 2000,
        timeout_s=120,
    )
    compose("kill", "-s", "SIGKILL", "consumer")
    compose("start", "consumer")
    thread.join()
    predictions = collect_predictions(set(produced["ids"]), timeout_s=600)
    assert len({p["event_id"] for p in predictions}) == 10_000
    assert len(predictions) == 10_000
