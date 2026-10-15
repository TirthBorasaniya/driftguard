"""Tests for the scoring consumer: idempotency leases, validation, DLQ, retries, batch flow."""

import ast
from pathlib import Path

import fakeredis
import httpx
import pytest

from driftguard.consumer.dlq import DeadLetterQueue, dlq_headers
from driftguard.consumer.idempotency import DONE, IdempotencyStore
from driftguard.consumer.main import BatchProcessor, Message
from driftguard.consumer.scoring import ScoringClient, ScoringUnavailableError
from driftguard.consumer.sinks import FeatureSink
from driftguard.consumer.validation import FlowValidator, ValidationError
from driftguard.data.registry import feature_names
from driftguard.data.synthetic import synthetic_flows
from driftguard.producer.replay import flow_record

pytestmark = pytest.mark.unit
FEATURES = feature_names("cicids2017")


def _records(n, seed=0):
    frame = synthetic_flows("cicids2017", n, seed=seed)
    return [flow_record(row, "cicids2017", FEATURES) for row in frame.to_dict("records")]


# ----- idempotency -----


def test_claim_then_done_then_duplicate():
    redis = fakeredis.FakeRedis(decode_responses=True)
    store = IdempotencyStore(redis, "a", ttl_s=172800, lease_s=30)
    first = store.claim(["e1", "e2"])
    assert first.new == ["e1", "e2"] and not first.duplicates
    store.mark_done(["e1", "e2"])
    assert redis.get("dg:seen:e1") == DONE
    assert 172000 < redis.ttl("dg:seen:e1") <= 172800
    again = store.claim(["e1", "e3"])
    assert again.duplicates == ["e1"] and again.new == ["e3"]


def test_held_claim_is_reclaimed_after_lease_expiry():
    redis = fakeredis.FakeRedis(decode_responses=True)
    dead = IdempotencyStore(redis, "dead", lease_s=30)
    alive = IdempotencyStore(redis, "alive", lease_s=30)
    dead.claim(["e1"])
    assert alive.claim(["e1"]).held == ["e1"]
    sleeps = []

    def expire(seconds):
        sleeps.append(seconds)
        redis.delete("dg:seen:e1")  # the dead owner's lease runs out

    result = alive.claim_with_wait(["e1"], sleep=expire, poll_s=1.0)
    assert result.new == ["e1"] and not result.held and sleeps == [1.0]


def test_held_claim_resolves_to_duplicate_when_owner_finishes():
    redis = fakeredis.FakeRedis(decode_responses=True)
    other = IdempotencyStore(redis, "other")
    me = IdempotencyStore(redis, "me")
    other.claim(["e1"])
    result = me.claim_with_wait(["e1"], sleep=lambda _s: other.mark_done(["e1"]), poll_s=0.1)
    assert result.duplicates == ["e1"] and not result.new


def test_release_only_removes_own_pending_claims():
    redis = fakeredis.FakeRedis(decode_responses=True)
    me, other = IdempotencyStore(redis, "me"), IdempotencyStore(redis, "other")
    me.claim(["a"])
    other.claim(["b"])
    me.release(["a", "b"])
    assert redis.get("dg:seen:a") is None and redis.get("dg:seen:b") is not None


# ----- validation and DLQ -----


def test_validator_accepts_valid_and_rejects_bad_values():
    validator = FlowValidator("cicids2017")
    record = _records(1)[0]
    validator.validate(record)
    for field, value, reason in [
        ("flow_duration", float("nan"), "finite"),
        ("dst_port", 70000.0, "above"),
        ("flow_duration", -5.0, "below"),
        ("event_id", "", "missing"),
    ]:
        bad = dict(record, **{field: value})
        with pytest.raises(ValidationError, match=reason):
            validator.validate(bad)


def test_dlq_headers_and_metric():
    sent = []
    dlq = DeadLetterQueue(lambda topic, **kw: sent.append((topic, kw)), "cicids2017")
    dlq.send(b"k", b"v", "decode", "bad magic", "flows.cicids2017.v1", 42)
    topic, kwargs = sent[0]
    assert topic == "flows.dlq" and kwargs["value"] == b"v"
    assert dict(kwargs["headers"]) == {
        "error_type": b"decode",
        "error_msg": b"bad magic",
        "source_topic": b"flows.cicids2017.v1",
        "source_offset": b"42",
    }
    with pytest.raises(ValueError):
        dlq_headers("other", "", "t", 0)


# ----- scoring client -----


def _scoring_client(handler):
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://serving")
    return ScoringClient(
        "http://serving", "cicids2017", client=client, sleep=lambda _s: None, seed=0
    )


def test_scoring_retries_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(503)
        body = __import__("json").loads(request.content)
        assert set(body) == {"dataset", "feature_names", "events"}
        results = [
            {
                "event_id": e["event_id"],
                "score": 0.1,
                "decision": 0,
                "threshold": 0.5,
                "model_alias": "champion",
                "model_version": "1",
            }
            for e in body["events"]
        ]
        return httpx.Response(
            200, json={"rollout_mode": "shadow", "results": results, "latency_ms": 1.0}
        )

    client = _scoring_client(handler)
    response = client.score([("e1", [1.0])], ["f"])
    assert len(calls) == 3 and response.results[0]["event_id"] == "e1"
    assert client.last_mode == "shadow"


def test_scoring_unavailable_after_retries():
    client = _scoring_client(lambda request: httpx.Response(500))
    with pytest.raises(ScoringUnavailableError):
        client.score([("e1", [1.0])], ["f"])


# ----- batch processor -----


class FakeScorer:
    def __init__(self, fail=False, mode="champion_only"):
        self.fail = fail
        self.last_mode = mode
        self.shadow_timeout_s = 1.0
        self.calls = 0

    def score(self, chunk, names):
        self.calls += 1
        if self.fail:
            raise ScoringUnavailableError("down")
        from driftguard.consumer.scoring import ScoreResponse

        return ScoreResponse(
            self.last_mode,
            [
                {
                    "event_id": eid,
                    "score": 0.9,
                    "decision": 1,
                    "threshold": 0.5,
                    "model_alias": "champion",
                    "model_version": "3",
                }
                for eid, _ in chunk
            ],
            2.0,
        )

    def score_shadow(self, chunk, names):
        return {"model_version": "4", "scores": {eid: 0.2 for eid, _ in chunk}}


def _processor(redis, scorer, dlq_sent, emitted, sink_frames):
    import json

    return BatchProcessor(
        dataset="cicids2017",
        deserialize=lambda payload: json.loads(payload),
        validator=FlowValidator("cicids2017"),
        idempotency=IdempotencyStore(redis, "test"),
        sink=FeatureSink("cicids2017", None, lambda name, frame: sink_frames.append(frame)),
        scorer=scorer,
        dlq=DeadLetterQueue(lambda topic, **kw: dlq_sent.append(kw), "cicids2017"),
        emit_prediction=emitted.append,
        flush=lambda: None,
        batch_size=4,
    )


def _messages(records, start_offset=0):
    import json

    return [
        Message(
            r["event_id"].encode(),
            json.dumps(r).encode(),
            "flows.cicids2017.v1",
            0,
            start_offset + i,
        )
        for i, r in enumerate(records)
    ]


def test_batch_dedupes_and_routes_bad_messages():
    redis = fakeredis.FakeRedis(decode_responses=True)
    dlq_sent, emitted, frames = [], [], []
    proc = _processor(redis, FakeScorer(), dlq_sent, emitted, frames)
    records = _records(10)
    bad_value = dict(records[1], flow_duration=-1.0)
    messages = _messages(records + [records[0]])
    messages.append(Message(b"x", b"not-json", "flows.cicids2017.v1", 0, 99))
    messages.append(_messages([bad_value], 100)[0])
    outcome = proc.process(messages)
    assert outcome.scored == 10 and outcome.duplicates == 1  # the in-batch copy of records[0]
    assert outcome.dlq == {"decode": 1, "validation": 1}
    assert len({p["event_id"] for p in emitted}) == 10 and len(emitted) == 10
    assert all(redis.get(f"dg:seen:{r['event_id']}") == DONE for r in records)
    assert "src_ip" not in frames[0].columns and "dst_ip" not in frames[0].columns
    # replaying the whole batch produces no new predictions
    replay = proc.process(_messages(records))
    assert replay.scored == 0 and replay.duplicates == 10 and len(emitted) == 10


def test_batch_sends_scoring_failures_to_dlq_and_releases_claims():
    redis = fakeredis.FakeRedis(decode_responses=True)
    dlq_sent, emitted, frames = [], [], []
    proc = _processor(redis, FakeScorer(fail=True), dlq_sent, emitted, frames)
    records = _records(5)
    outcome = proc.process(_messages(records))
    assert outcome.dlq == {"scoring_unavailable": 5} and not emitted
    assert all(redis.get(f"dg:seen:{r['event_id']}") is None for r in records)


def test_shadow_scores_attached_in_shadow_mode():
    redis = fakeredis.FakeRedis(decode_responses=True)
    dlq_sent, emitted, frames = [], [], []
    proc = _processor(redis, FakeScorer(mode="shadow"), dlq_sent, emitted, frames)
    proc.process(_messages(_records(3)))
    assert all(p["shadow_score"] == 0.2 and p["shadow_version"] == "4" for p in emitted)
    assert all(p["model_version"] == "3" for p in emitted)


def test_consumer_package_has_no_training_imports():
    root = Path(__file__).resolve().parents[2] / "src" / "driftguard" / "consumer"
    forbidden = ("driftguard.training", "lightgbm", "prefect", "driftguard.benchmark")
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert not name.startswith(forbidden), f"{path.name} imports {name}"
