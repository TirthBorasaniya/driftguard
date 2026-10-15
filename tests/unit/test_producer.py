"""Tests for the replay producer: plan determinism, label delay, coverage, duplicates, pacing."""

from collections import Counter

import pytest

from driftguard.data.synthetic import synthetic_flows
from driftguard.producer.labels import DUPLICATE, FLOW, LABEL, PlanConfig, emission_plan
from driftguard.producer.rate_limit import RealtimePacer, TokenBucket
from driftguard.producer.replay import flow_record, plan_config_from, run_replay

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_plan_is_deterministic():
    config = PlanConfig(label_delay_min=5, label_delay_max=20, duplicate_rate=0.1, seed=7)
    assert list(emission_plan(500, config)) == list(emission_plan(500, config))
    other = PlanConfig(label_delay_min=5, label_delay_max=20, duplicate_rate=0.1, seed=8)
    assert list(emission_plan(500, config)) != list(emission_plan(500, other))


def test_label_delay_within_bounds_and_full_coverage():
    config = PlanConfig(label_delay_min=10, label_delay_max=30, label_coverage=1.0, seed=1)
    flow_pos, label_pos = {}, {}
    flows_seen = 0
    for kind, index in emission_plan(1000, config):
        if kind == FLOW:
            flows_seen += 1
            flow_pos[index] = flows_seen
        elif kind == LABEL:
            label_pos[index] = flows_seen
    assert set(label_pos) == set(range(1000))
    for index, seen in label_pos.items():
        if index + 30 < 1000:
            delay = seen - flow_pos[index]
            assert 10 <= delay <= 30


def test_label_coverage_fraction():
    config = PlanConfig(label_delay_min=1, label_delay_max=2, label_coverage=0.3, seed=3)
    counts = Counter(kind for kind, _ in emission_plan(20000, config))
    assert counts[FLOW] == 20000
    assert abs(counts[LABEL] / 20000 - 0.3) < 0.02


def test_duplicate_injection_rate():
    config = PlanConfig(label_delay_min=1, label_delay_max=2, duplicate_rate=0.05, seed=4)
    counts = Counter(kind for kind, _ in emission_plan(20000, config))
    assert abs(counts[DUPLICATE] / 20000 - 0.05) < 0.01


def test_invalid_plan_rejected():
    with pytest.raises(ValueError):
        PlanConfig(label_delay_min=10, label_delay_max=5)


def test_token_bucket_rate():
    clock = FakeClock()
    bucket = TokenBucket(rate=100, burst=1, clock=clock, sleep=clock.sleep)
    for _ in range(101):
        bucket.acquire()
    assert clock.now == pytest.approx(1.0, rel=1e-6)


def test_realtime_pacer_scales_event_time():
    clock = FakeClock()
    pacer = RealtimePacer(factor=10, clock=clock, sleep=clock.sleep)
    pacer.wait_until(1_000)
    pacer.wait_until(11_000)
    assert clock.now == pytest.approx(1.0)


def test_flow_record_has_no_label():
    frame = synthetic_flows("cicids2017", 3, seed=1)
    record = flow_record(frame.iloc[0].to_dict(), "cicids2017", ["flow_duration"])
    assert "label" not in record and "label_family" not in record
    assert set(record) == {
        "event_id",
        "dataset",
        "source_file",
        "row_index",
        "event_time",
        "src_ip",
        "dst_ip",
        "flow_duration",
    }


def test_run_replay_emits_identical_sequences():
    frame = synthetic_flows("cicids2017", 300, seed=2)
    config = {
        "speed": {"mode": "max"},
        "label_delay_min": 5,
        "label_delay_max": 10,
        "label_coverage": 1.0,
        "duplicate_rate": 0.05,
        "seed": 11,
        "flush_every": 100,
    }

    def run():
        sent = []
        stats = run_replay(
            frame,
            "cicids2017",
            config,
            send=lambda topic, key, value: sent.append((topic, key, value)),
            serialize_flow=lambda r: repr(sorted(r.items())).encode(),
            serialize_label=lambda r: repr(sorted(r.items())).encode(),
            flow_topic="flows",
            label_topic="labels",
        )
        return sent, stats

    first, stats = run()
    second, _ = run()
    assert first == second
    assert stats.flows == 300 and stats.labels == 300 and stats.duplicates > 0


def test_profile_overrides_coverage():
    config = {
        "label_delay_min": 1,
        "label_delay_max": 2,
        "label_coverage": 1.0,
        "duplicate_rate": 0.0,
        "seed": 1,
        "profile": "realistic",
        "profiles": {"realistic": {"label_coverage": 0.3}},
    }
    assert plan_config_from(config).label_coverage == 0.3
