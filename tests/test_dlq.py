"""Tests for dead letter queue routing (DLQ hardening, infra improvement 1)."""

import pytest

confluent_kafka = pytest.importorskip("confluent_kafka")


class FakeProducer:
    """Records produce() calls without a live Kafka connection."""

    def __init__(self):
        self.produced = []

    def produce(self, topic, value, headers):
        self.produced.append({"topic": topic, "value": value, "headers": headers})

    def poll(self, timeout):
        return 0


def test_route_to_dlq_publishes_validation_error_header():
    from src.consumer.dlq_handler import DLQ_ERROR_HEADER_KEY, DLQ_TOPIC, route_to_dlq

    producer = FakeProducer()
    raw_message = b'{"bad": "payload"}'
    validation_error = "field required: event_id"

    route_to_dlq(producer, raw_message, validation_error)

    assert len(producer.produced) == 1
    record = producer.produced[0]
    assert record["topic"] == DLQ_TOPIC
    assert record["value"] == raw_message
    assert DLQ_ERROR_HEADER_KEY in record["headers"]
    assert record["headers"][DLQ_ERROR_HEADER_KEY] == validation_error.encode("utf-8")
