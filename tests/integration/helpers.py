"""Helpers for tests that run against the Docker Compose stack (never run in unit CI)."""

import re
import subprocess
import time
import uuid
from collections.abc import Callable

import httpx

from driftguard.common.kafka import consumer_config, producer_config
from driftguard.common.settings import get_settings
from driftguard.data.registry import feature_names
from driftguard.data.schemas import topic_name
from driftguard.data.serde import TopicSerde, registry_client
from driftguard.data.synthetic import synthetic_flows
from driftguard.producer.replay import run_replay

# ============= Constants =============

DATASET = "cicids2017"
METRIC_LINE = re.compile(r"^(?P<name>[a-z_]+)(\{(?P<labels>[^}]*)\})? (?P<value>[0-9.eE+-]+)$")


# ============= Stack =============


def compose(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run `docker compose <args>` from the repository root."""
    return subprocess.run(["docker", "compose", *args], check=check, capture_output=True, text=True)


def wait_for(predicate: Callable[[], bool], timeout_s: float, interval_s: float = 1.0) -> bool:
    """Poll until `predicate()` is true or the timeout elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def metric_value(url: str, name: str, **labels: str) -> float:
    """Sum a Prometheus metric from a `/metrics` endpoint, filtered by labels."""
    total = 0.0
    for line in httpx.get(url, timeout=5).text.splitlines():
        match = METRIC_LINE.match(line)
        if not match or match.group("name") != name:
            continue
        found = dict(re.findall(r'(\w+)="([^"]*)"', match.group("labels") or ""))
        if all(found.get(k) == v for k, v in labels.items()):
            total += float(match.group("value"))
    return total


# ============= Producing and collecting =============


def produce_synthetic(
    n_events: int, duplicate_rate: float, rate_eps: float | None = None, seed: int = 1
) -> tuple[list[str], dict]:
    """Produce synthetic flows (unique ids per call) to the flow topic; returns ids and stats."""
    from confluent_kafka import Producer

    settings = get_settings()
    tag = uuid.uuid4().hex[:8]
    frame = synthetic_flows(DATASET, n_events, seed=seed, id_tag=f"-{tag}")
    client = registry_client(settings.schema_registry_url)
    flows, labels = TopicSerde(client, "flows", DATASET), TopicSerde(client, "labels", DATASET)
    producer = Producer(producer_config(settings.kafka_bootstrap_servers, f"it-{tag}"))
    speed = (
        {"mode": "fixed_rate", "rate_eps": rate_eps, "burst": rate_eps}
        if rate_eps
        else {"mode": "max"}
    )
    config = {
        "speed": speed,
        "label_delay_min": 100,
        "label_delay_max": 500,
        "label_coverage": 1.0,
        "duplicate_rate": duplicate_rate,
        "seed": seed,
        "flush_every": 1000,
    }

    def send(topic, key, value):
        producer.produce(topic, key=key.encode(), value=value)
        producer.poll(0)

    stats = run_replay(
        frame,
        DATASET,
        config,
        send,
        flows.serialize,
        labels.serialize,
        flows.topic,
        labels.topic,
        flush=lambda: producer.flush(30),
    )
    return frame["event_id"].tolist(), stats.as_dict()


def collect_predictions(event_ids: set[str], timeout_s: float = 300.0) -> list[dict]:
    """Read `predictions.v1` from the beginning with a fresh group; return records for the ids."""
    from confluent_kafka import Consumer

    settings = get_settings()
    serde = TopicSerde(registry_client(settings.schema_registry_url), "predictions")
    consumer = Consumer(
        consumer_config(
            settings.kafka_bootstrap_servers, f"it-collect-{uuid.uuid4().hex}", "it-collect"
        )
    )
    consumer.subscribe([topic_name("predictions")])
    found, seen = [], set()
    deadline, idle_since = time.monotonic() + timeout_s, time.monotonic()
    while time.monotonic() < deadline:
        messages = consumer.consume(num_messages=5000, timeout=1.0)
        if not messages and seen >= event_ids and time.monotonic() - idle_since > 15:
            break
        for message in messages:
            if message.error():
                continue
            idle_since = time.monotonic()
            record = serde.deserialize(message.value())
            if record["event_id"] in event_ids:
                found.append(record)
                seen.add(record["event_id"])
    consumer.close()
    return found


def collect_dlq(source_offsets_after: int | None = None, timeout_s: float = 60.0) -> list[dict]:
    """Read the DLQ from the beginning; return headers of each record."""
    from confluent_kafka import Consumer

    settings = get_settings()
    consumer = Consumer(
        consumer_config(settings.kafka_bootstrap_servers, f"it-dlq-{uuid.uuid4().hex}", "it-dlq")
    )
    consumer.subscribe([topic_name("dlq")])
    records, deadline = [], time.monotonic() + timeout_s
    idle = 0
    while time.monotonic() < deadline and idle < 10:
        messages = consumer.consume(num_messages=1000, timeout=1.0)
        idle = 0 if messages else idle + 1
        for message in messages:
            if message.error():
                continue
            headers = {k: v.decode() for k, v in (message.headers() or [])}
            headers["key"] = (message.key() or b"").decode(errors="replace")
            records.append(headers)
    consumer.close()
    return records


def features() -> list[str]:
    """Flow feature names."""
    return feature_names(DATASET)
