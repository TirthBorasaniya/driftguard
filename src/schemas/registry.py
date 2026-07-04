"""Confluent Schema Registry integration: schema registration and Avro serde helpers."""

from pathlib import Path

from confluent_kafka.schema_registry import Schema, SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer, AvroSerializer

from src.config import PROJECT_ROOT, settings

SCHEMA_REGISTRY_URL = settings.schema_registry_url
SUBJECT_NAME = "network_flows-value"
SCHEMA_PATH = PROJECT_ROOT / "src" / "schemas" / "network_flow_event.avsc"


def register_schema(
    schema_registry_url: str = SCHEMA_REGISTRY_URL,
    subject_name: str = SUBJECT_NAME,
    schema_path: str = str(SCHEMA_PATH),
) -> int:
    """
    Register the Avro schema with the Schema Registry on startup.

    Parameters
    ----------
    schema_registry_url : str
        Base URL of the Schema Registry service.
    subject_name : str
        Subject name under which to register the schema.
    schema_path : str
        Path to the local .avsc schema file.

    Returns
    -------
    schema_id : int
        The registered schema ID returned by the registry.
    """
    schema_str = Path(schema_path).read_text()
    client = SchemaRegistryClient({"url": schema_registry_url})
    schema = Schema(schema_str, schema_type="AVRO")
    schema_id = client.register_schema(subject_name, schema)
    return schema_id


def build_avro_serializer(
    schema_registry_url: str = SCHEMA_REGISTRY_URL,
    schema_path: str = str(SCHEMA_PATH),
) -> AvroSerializer:
    """
    Build a registry-aware Avro serializer for producing NetworkFlowEvent messages.

    Parameters
    ----------
    schema_registry_url : str
        Base URL of the Schema Registry service.
    schema_path : str
        Path to the local .avsc schema file (source of the schema string only;
        the serializer itself resolves and caches schema IDs via the registry).

    Returns
    -------
    avro_serializer : AvroSerializer
    """
    schema_str = Path(schema_path).read_text()
    client = SchemaRegistryClient({"url": schema_registry_url})
    return AvroSerializer(client, schema_str)


def build_avro_deserializer(
    schema_registry_url: str = SCHEMA_REGISTRY_URL,
    schema_path: str = str(SCHEMA_PATH),
) -> AvroDeserializer:
    """
    Build a registry-aware Avro deserializer for consuming NetworkFlowEvent messages.

    Parameters
    ----------
    schema_registry_url : str
        Base URL of the Schema Registry service.
    schema_path : str
        Path to the local .avsc schema file.

    Returns
    -------
    avro_deserializer : AvroDeserializer
    """
    schema_str = Path(schema_path).read_text()
    client = SchemaRegistryClient({"url": schema_registry_url})
    return AvroDeserializer(client, schema_str)
