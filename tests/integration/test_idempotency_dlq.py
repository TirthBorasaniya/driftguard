"""Integration: duplicates yield exactly one prediction each; malformed messages reach the DLQ."""

import uuid

import pytest

from driftguard.common.kafka import producer_config
from driftguard.common.settings import get_settings
from driftguard.data.schemas import topic_name
from driftguard.data.serde import TopicSerde, registry_client
from driftguard.data.synthetic import synthetic_flows
from driftguard.producer.replay import flow_record

from .helpers import (
    DATASET,
    collect_dlq,
    collect_predictions,
    features,
    metric_value,
    produce_synthetic,
)

pytestmark = pytest.mark.integration
CONSUMER_METRICS = "http://localhost:9101/metrics"


def test_10000_events_with_5pct_duplicates_give_exactly_10000_predictions():
    before = metric_value(CONSUMER_METRICS, "driftguard_duplicates_total", dataset=DATASET)
    event_ids, stats = produce_synthetic(10_000, duplicate_rate=0.05)
    assert stats["duplicates"] > 0
    predictions = collect_predictions(set(event_ids))
    assert len(predictions) == 10_000
    assert len({p["event_id"] for p in predictions}) == 10_000
    after = metric_value(CONSUMER_METRICS, "driftguard_duplicates_total", dataset=DATASET)
    assert after - before == stats["duplicates"]


def test_100_malformed_messages_reach_dlq_with_error_type():
    from confluent_kafka import Producer

    settings = get_settings()
    tag = uuid.uuid4().hex[:8]
    producer = Producer(producer_config(settings.kafka_bootstrap_servers, f"it-dlq-{tag}"))
    topic = topic_name("flows", DATASET)
    for i in range(50):
        producer.produce(topic, key=f"decode-{tag}-{i}".encode(), value=b"\x00not-avro")
    serde = TopicSerde(registry_client(settings.schema_registry_url), "flows", DATASET)
    frame = synthetic_flows(DATASET, 50, seed=3, id_tag=f"-dlq-{tag}")
    for i, row in enumerate(frame.to_dict("records")):
        record = flow_record(row, DATASET, features())
        record["flow_duration"] = -1.0  # violates the physical lower bound
        producer.produce(topic, key=f"validation-{tag}-{i}".encode(), value=serde.serialize(record))
    producer.flush(30)
    records = [r for r in collect_dlq() if tag in r["key"]]
    assert len(records) == 100
    types = sorted(r["error_type"] for r in records)
    assert types.count("decode") == 50 and types.count("validation") == 50
    assert all(r["source_topic"] == topic and r["source_offset"].isdigit() for r in records)
