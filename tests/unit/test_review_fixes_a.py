"""Regression tests for review findings H2, H3, M3, M4, M7, and M10."""

import json

import fakeredis
import pandas as pd
import pytest

from driftguard.common.kafka import DeliveryError, DeliveryTracker
from driftguard.consumer.idempotency import DONE, HeldClaimsError, IdempotencyStore
from driftguard.consumer.main import Message, behavior_sink_enabled
from driftguard.consumer.sinks import LabelSink
from driftguard.monitor.main import PAUSE_KEY, PAUSED_BY_DATA_QUALITY

from .test_consumer import FakeScorer, _messages, _processor, _records
from .test_monitor import _core

pytestmark = pytest.mark.unit


class FakeMsg:
    def topic(self):
        return "predictions.v1"

    def partition(self):
        return 0


# ----- H2: delivery failures must stop mark_done and the offset commit -----


def test_failed_delivery_aborts_batch_before_done_marking():
    redis = fakeredis.FakeRedis(decode_responses=True)
    tracker = DeliveryTracker()

    def emit(prediction):
        tracker("Broker: Message size too large", FakeMsg())  # delivery report with an error

    dlq_sent, frames = [], []
    proc = _processor(redis, FakeScorer(), dlq_sent, [], frames)
    proc.emit_prediction = emit
    proc.flush = tracker.raise_if_failed  # what ConsumerService._flush does after producer.flush
    records = _records(5)
    with pytest.raises(DeliveryError):
        proc.process(_messages(records))
    for record in records:  # none may be marked done, so redelivery rescored them
        assert redis.get(f"dg:seen:{record['event_id']}") != DONE


def test_drift_event_delivery_failure_is_raised():
    from driftguard.monitor.main import MonitorService

    class Producer:
        def produce(self, topic, key, value, on_delivery):
            on_delivery("fail", FakeMsg())

        def flush(self, timeout):
            return 0

    class Serde:
        topic = "drift.events"

        def serialize(self, record):
            return b"x"

    service = object.__new__(MonitorService)
    service.producer, service.drift_serde, service.dataset = Producer(), Serde(), "cicids2017"
    service.delivery = DeliveryTracker()
    event = {
        "window_id": "w",
        "severity": "LOW",
        "signals": {},
        "window_end_ms": 1,
        "dataset": "d",
        "detected_at": 1,
        "policy_version": 3,
    }
    with pytest.raises(DeliveryError):
        service.publish_drift_events([event])


# ----- H3: an operator pause survives data-quality failures -----


def test_operator_pause_survives_gx_failure_then_clean_window():
    core, _, redis = _core()
    key = PAUSE_KEY.format(dataset="cicids2017")
    redis.set(key, "operator:e4-static")
    core._apply_pause(False)
    assert redis.get(key) == "operator:e4-static"
    core._apply_pause(True)
    assert redis.get(key) == "operator:e4-static"


def test_data_quality_pause_is_set_and_lifted():
    core, _, redis = _core()
    key = PAUSE_KEY.format(dataset="cicids2017")
    core._apply_pause(False)
    assert redis.get(key) == PAUSED_BY_DATA_QUALITY
    core._apply_pause(True)
    assert redis.get(key) is None


# ----- M10: leases are renewed and held ids fail the batch -----


def test_renew_extends_only_own_pending_claims():
    redis = fakeredis.FakeRedis(decode_responses=True)
    mine = IdempotencyStore(redis, "me", lease_s=120)
    other = IdempotencyStore(redis, "other", lease_s=120)
    mine.claim(["a"])
    other.claim(["b"])
    redis.expire("dg:seen:a", 5)
    redis.expire("dg:seen:b", 5)
    mine.renew(["a", "b"])
    assert redis.ttl("dg:seen:a") > 100 and redis.ttl("dg:seen:b") <= 5


def test_batch_fails_when_ids_stay_held_by_another_consumer():
    redis = fakeredis.FakeRedis(decode_responses=True)
    records = _records(3)
    IdempotencyStore(redis, "someone-else", lease_s=120).claim([r["event_id"] for r in records])
    dlq_sent, emitted, frames = [], [], []
    proc = _processor(redis, FakeScorer(), dlq_sent, emitted, frames)
    proc.idempotency.lease_s = 1  # the wait budget is twice the lease: two polls
    proc.wait_poll_s = 0.01
    with pytest.raises(HeldClaimsError):
        proc.process(_messages(records))
    assert not emitted


def test_lease_is_renewed_between_chunks():
    redis = fakeredis.FakeRedis(decode_responses=True)
    dlq_sent, emitted, frames = [], [], []
    proc = _processor(redis, FakeScorer(), dlq_sent, emitted, frames)
    renewals = []
    original = proc.idempotency.renew
    proc.idempotency.renew = lambda ids: (renewals.append(len(ids)), original(ids))[1]
    proc.process(_messages(_records(10)))  # batch_size is 4 in the helper: three chunks
    assert len(renewals) == 3


# ----- M4: label sink write failures keep the buffer and do not commit -----


class FlakyOffline:
    def __init__(self):
        self.calls = 0
        self.rows = []

    def append(self, dataset, table, frame):
        self.calls += 1
        if self.calls == 1:
            raise OSError("disk full")
        self.rows.append(frame)


class FakeConsumer:
    def __init__(self):
        self.commits = 0

    def commit(self, asynchronous=False):
        self.commits += 1


def test_label_sink_retries_failed_write_without_commit_or_loss():
    offline, consumer = FlakyOffline(), FakeConsumer()
    sink = LabelSink(consumer, json.loads, offline, "cicids2017")
    sink._buffer = [{"event_id": "e1", "label": 1, "label_family": "DDoS", "label_time": 1000}]
    assert sink._try_flush() is False
    assert consumer.commits == 0 and len(sink._buffer) == 1 and sink.write_failures == 1
    assert sink._try_flush() is True
    assert consumer.commits == 1 and sink._buffer == [] and len(offline.rows) == 1
    assert isinstance(offline.rows[0], pd.DataFrame)


# ----- M3: a poison record does not end the rollout controller -----


def test_rollout_service_skips_undecodable_records():
    from driftguard.orchestrator.rollout import RolloutService

    class Message:
        def __init__(self, topic, value):
            self._t, self._v = topic, value

        def error(self):
            return None

        def topic(self):
            return self._t

        def value(self):
            return self._v

    class Consumer:
        def __init__(self, service):
            self.service, self.polls = service, 0

        def consume(self, num_messages, timeout):
            self.polls += 1
            if self.polls == 1:
                return [Message("predictions.v1", b"poison")]
            self.service._running = False
            return []

        def commit(self, asynchronous=True):
            pass

        def close(self):
            pass

    class Serde:
        def deserialize(self, payload):
            raise ValueError("bad magic byte")

    class Controller:
        def on_predictions(self, records):
            pass

        def on_labels(self, records):
            pass

        def step(self):
            return None

    service = object.__new__(RolloutService)
    service.dataset, service._running, service.decode_failures = "cicids2017", True, 0
    service.topics = {"predictions.v1": Serde()}
    service.controller, service.consumer = Controller(), Consumer(service)
    service.run()
    assert service.decode_failures == 1


# ----- M7: load-test off mode never writes behavioral rows -----


def test_behavior_sink_disabled_in_off_mode():
    assert behavior_sink_enabled("cicids2017", "on") is True
    assert behavior_sink_enabled("cicids2017", "off") is False
    assert behavior_sink_enabled("kyoto", "on") is False


def test_message_dataclass_unchanged():
    assert Message(b"k", b"v", "t", 0, 1).offset == 1
