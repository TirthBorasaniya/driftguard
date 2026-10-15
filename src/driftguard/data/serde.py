"""Avro serialization over Confluent Schema Registry with fixed subjects (SPEC 3.1)."""

import json
from datetime import datetime
from typing import Any

from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer
from confluent_kafka.schema_registry._sync.mock_schema_registry_client import MockSchemaRegistryClient
from confluent_kafka.schema_registry.schema_registry_client import Schema
from confluent_kafka.serialization import MessageField, SerializationContext

from driftguard.data.schemas import load_schema, subject_name, topic_name

# ============= Constants =============

COMPATIBILITY = "BACKWARD"
TIMESTAMP_FIELDS = ("event_time", "label_time", "detected_at")


# ============= Registry =============


def registry_client(url: str) -> SchemaRegistryClient:
    """Create a Schema Registry client; `mock://` URLs return an in-memory client for tests."""
    return SchemaRegistryClient.new_client({"url": url})


def register_subject(client: SchemaRegistryClient, subject: str) -> int:
    """
    Register the committed schema for a subject and set its compatibility to BACKWARD.

    Parameters
    ----------
    client : SchemaRegistryClient
        Registry client.
    subject : str
        Subject name, for example `flows-cicids2017-value`.

    Returns
    -------
    schema_id : int
        Registry id of the registered schema.
    """
    schema = Schema(json.dumps(load_schema(subject)), schema_type="AVRO")
    schema_id = client.register_schema(subject, schema)
    # the in-memory test client has no compatibility endpoint
    if not isinstance(client, MockSchemaRegistryClient):
        client.set_compatibility(subject_name=subject, level=COMPATIBILITY)
    return schema_id


# ============= Serde =============


def _fixed_subject(subject: str):
    """Subject name strategy that always returns one subject regardless of topic."""

    def strategy(_ctx: SerializationContext | None, _record_name: str | None) -> str:
        """Return the fixed subject name."""
        return subject

    return strategy


def _to_epoch_ms(record: dict[str, Any]) -> dict[str, Any]:
    """Convert timestamp-millis fields decoded as datetimes back to epoch milliseconds."""
    for field in TIMESTAMP_FIELDS:
        value = record.get(field)
        if isinstance(value, datetime):
            record[field] = int(value.timestamp() * 1000)
    return record


class TopicSerde:
    """
    Serializer and deserializer pair for one stream kind.

    Parameters
    ----------
    client : SchemaRegistryClient
        Registry client.
    kind : str
        Stream kind (`flows`, `labels`, `predictions`, `drift`).
    dataset : str or None
        Dataset for dataset-scoped kinds.
    """

    def __init__(self, client: SchemaRegistryClient, kind: str, dataset: str | None = None):
        self.kind = kind
        self.topic = topic_name(kind, dataset)
        self.subject = subject_name(kind, dataset)
        schema_str = json.dumps(load_schema(self.subject))
        self._serializer = AvroSerializer(
            schema_registry_client=client,
            schema_str=schema_str,
            conf={
                "auto.register.schemas": True,
                "subject.name.strategy": _fixed_subject(self.subject),
            },
        )
        self._deserializer = AvroDeserializer(schema_registry_client=client)
        self._ctx = SerializationContext(self.topic, MessageField.VALUE)

    def serialize(self, record: dict[str, Any]) -> bytes:
        """Encode a record dict to Confluent wire-format Avro bytes."""
        return self._serializer(record, self._ctx)

    def deserialize(self, payload: bytes) -> dict[str, Any]:
        """Decode Confluent wire-format Avro bytes into a record dict with epoch-ms timestamps."""
        record = self._deserializer(payload, self._ctx)
        if not isinstance(record, dict):
            raise ValueError(f"decoded {self.kind} payload is not a record")
        return _to_epoch_ms(record)
