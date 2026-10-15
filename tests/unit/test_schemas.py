"""Tests for Avro schemas: committed files match generation and round-trip through the serde."""

import pytest

from driftguard.data.registry import feature_names
from driftguard.data.schemas import (
    all_schemas,
    load_schema,
    render_schema,
    schema_path,
    subject_name,
    topic_name,
)
from driftguard.data.serde import TopicSerde, register_subject, registry_client

pytestmark = pytest.mark.unit


def test_committed_schemas_match_generation():
    for subject, schema in all_schemas(("cicids2017", "kyoto")).items():
        assert schema_path(subject).read_text() == render_schema(schema), subject


def test_flow_schema_has_no_label_and_all_features():
    fields = [f["name"] for f in load_schema("flows-cicids2017-value")["fields"]]
    assert "label" not in fields
    assert fields[-len(feature_names("cicids2017")) :] == feature_names("cicids2017")


def test_topic_and_subject_names():
    assert topic_name("flows", "kyoto") == "flows.kyoto.v1"
    assert topic_name("predictions") == "predictions.v1"
    assert subject_name("flows", "cicids2017") == "flows-cicids2017-value"
    assert subject_name("drift") == "drift-events-value"


def test_serde_roundtrip_with_mock_registry():
    client = registry_client("mock://driftguard-test")
    assert register_subject(client, "labels-value") > 0
    serde = TopicSerde(client, "labels", "cicids2017")
    record = {"event_id": "e1", "label": 1, "label_family": "DDoS", "label_time": 1499083200123}
    payload = serde.serialize(record)
    assert payload[0] == 0  # Confluent wire format magic byte
    assert serde.deserialize(payload) == record


def test_prediction_roundtrip_nullable_shadow():
    client = registry_client("mock://driftguard-test-2")
    serde = TopicSerde(client, "predictions")
    record = {
        "event_id": "e",
        "dataset": "cicids2017",
        "event_time": 1,
        "model_alias": "champion",
        "model_version": "3",
        "score": 0.5,
        "decision": 1,
        "threshold": 0.2,
        "latency_ms": 1.5,
        "shadow_score": None,
        "shadow_version": None,
    }
    assert serde.deserialize(serde.serialize(record)) == record
