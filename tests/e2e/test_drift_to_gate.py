"""E2E: injected covariate shift -> drift event within 5 windows -> retraining -> shadow -> gate."""

import json
import time
import uuid

import pytest

from driftguard.benchmark.injector import inject
from driftguard.common.config import load_config
from driftguard.common.kafka import consumer_config, producer_config
from driftguard.common.mlflow_client import CHALLENGER, MlflowRegistry, model_name
from driftguard.common.settings import REPO_ROOT, get_settings
from driftguard.data.schemas import topic_name
from driftguard.data.serde import TopicSerde, registry_client
from driftguard.data.synthetic import synthetic_flows
from driftguard.producer.replay import run_replay
from tests.integration.helpers import DATASET, wait_for

pytestmark = [pytest.mark.e2e, pytest.mark.slow]
WINDOWS_ALLOWED = 5
STEP_MS = 5


def drift_events(group: str, timeout_s: float) -> list[dict]:
    """Read drift events from the beginning with a fresh group."""
    from confluent_kafka import Consumer

    settings = get_settings()
    serde = TopicSerde(registry_client(settings.schema_registry_url), "drift")
    consumer = Consumer(consumer_config(settings.kafka_bootstrap_servers, group, "e2e"))
    consumer.subscribe([topic_name("drift")])
    events, deadline = [], time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for message in consumer.consume(num_messages=100, timeout=1.0):
            if not message.error():
                events.append(serde.deserialize(message.value()))
    consumer.close()
    return events


def test_injected_shift_flows_through_detection_retraining_shadow_and_gate():
    from confluent_kafka import Producer
    from redis import Redis

    settings = get_settings()
    policy = load_config(REPO_ROOT / "configs" / "drift_policy.yaml").data
    window = int(policy["window_size"])
    tag = uuid.uuid4().hex[:8]
    n_windows, onset_window = 24, 8
    frame = synthetic_flows(
        DATASET,
        n_windows * window,
        seed=21,
        id_tag=f"-e2e-{tag}",
        start_time_ms=int(time.time() * 1000),
        step_ms=STEP_MS,
    )
    shifted, manifest = inject(
        frame,
        "covariate",
        onset_window * window,
        {
            "features": ["flow_duration", "flow_bytes_per_sec", "flow_iat_mean"],
            "factor": 2.0,
            "noise_std": 0.1,
        },
        seed=3,
        reference=frame,
    )
    client = registry_client(settings.schema_registry_url)
    flows, labels = TopicSerde(client, "flows", DATASET), TopicSerde(client, "labels", DATASET)
    producer = Producer(producer_config(settings.kafka_bootstrap_servers, f"e2e-{tag}"))
    config = {
        "speed": {"mode": "fixed_rate", "rate_eps": 1000, "burst": 1000},
        "label_delay_min": 200,
        "label_delay_max": 1000,
        "label_coverage": 1.0,
        "duplicate_rate": 0.0,
        "seed": 5,
        "flush_every": 2000,
    }
    started = time.time()
    run_replay(
        shifted,
        DATASET,
        config,
        lambda t, k, v: (producer.produce(t, key=k.encode(), value=v), producer.poll(0)),
        flows.serialize,
        labels.serialize,
        flows.topic,
        labels.topic,
        flush=lambda: producer.flush(30),
    )

    events = [
        e for e in drift_events(f"e2e-drift-{tag}", 120) if e["detected_at"] >= started * 1000
    ]
    assert events, "no drift event emitted"
    # monitor window ids are global, so delay is measured in event time from the injected onset
    onset_time = int(shifted["event_time"].iloc[manifest.onset])
    window_span_ms = window * STEP_MS
    delay_windows = (events[0]["signals"]["window_end_ms"] - onset_time) / window_span_ms
    assert 0 <= delay_windows <= WINDOWS_ALLOWED, events[0]

    registry = MlflowRegistry(settings.mlflow_tracking_uri)
    assert wait_for(
        lambda: registry.get_alias(model_name(DATASET), CHALLENGER) is not None,
        timeout_s=900,
        interval_s=10,
    ), "no challenger registered by retraining"
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    assert wait_for(
        lambda: redis.get(f"dg:rollout:{DATASET}") in ("shadow", "canary", "champion_only"), 60
    )

    log = settings.promotions_log_path

    def gate_decided() -> bool:
        if not log.exists():
            return False
        entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        return any(
            e["action"] in ("promote_to_canary", "decline")
            and e["ts"] >= time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(started))
            for e in entries
        )

    assert wait_for(gate_decided, timeout_s=1800, interval_s=15), "no gate decision recorded"
