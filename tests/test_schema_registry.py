"""Tests for Confluent Schema Registry integration (infra improvement 7)."""

import urllib.request

import pytest

pytest.importorskip("confluent_kafka.schema_registry")

from src.schemas.registry import (  # noqa: E402
    SCHEMA_PATH,
    SCHEMA_REGISTRY_URL,
    SUBJECT_NAME,
    register_schema,
)


def _registry_reachable() -> bool:
    try:
        urllib.request.urlopen(f"{SCHEMA_REGISTRY_URL}/subjects", timeout=1)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _registry_reachable(), reason="requires a running Schema Registry instance")
def test_register_schema_returns_positive_schema_id():
    schema_id = register_schema(
        schema_registry_url=SCHEMA_REGISTRY_URL,
        subject_name=SUBJECT_NAME,
        schema_path=str(SCHEMA_PATH),
    )
    assert isinstance(schema_id, int)
    assert schema_id > 0
