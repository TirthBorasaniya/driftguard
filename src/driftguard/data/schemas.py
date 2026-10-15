"""Avro schema generation and loading for every DriftGuard topic (SPEC 3.1)."""

import json
from pathlib import Path

from driftguard.data.registry import feature_names

# ============= Constants =============

SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
NAMESPACE = "io.driftguard"
TIMESTAMP_MILLIS = {"type": "long", "logicalType": "timestamp-millis"}

TOPICS = {
    "flows": "flows.{dataset}.v1",
    "labels": "labels.{dataset}.v1",
    "predictions": "predictions.v1",
    "drift": "drift.events",
    "dlq": "flows.dlq",
}
SUBJECTS = {
    "flows": "flows-{dataset}-value",
    "labels": "labels-value",
    "predictions": "predictions-value",
    "drift": "drift-events-value",
}
PARTITIONS = {"flows": 6, "labels": 6, "predictions": 6, "drift": 1, "dlq": 1}
# datasets whose flow messages carry raw IPs for stream features (never model features)
DATASETS_WITH_IPS = {"cicids2017"}


# ============= Names =============


def topic_name(kind: str, dataset: str | None = None) -> str:
    """Return the topic for a stream kind (`flows`, `labels`, `predictions`, `drift`, `dlq`)."""
    return TOPICS[kind].format(dataset=dataset)


def subject_name(kind: str, dataset: str | None = None) -> str:
    """Return the Schema Registry subject for a stream kind."""
    return SUBJECTS[kind].format(dataset=dataset)


# ============= Schema builders =============


def flow_schema(dataset: str) -> dict:
    """
    Build the flow value schema for a dataset.

    Flow messages carry identifiers, event time, the allowlisted features as doubles, and, for
    CICIDS2017 only, raw IPs used solely to derive salted stream-feature keys. They carry no label.
    """
    fields = [
        {"name": "event_id", "type": "string"},
        {"name": "dataset", "type": "string"},
        {"name": "source_file", "type": "string"},
        {"name": "row_index", "type": "long"},
        {"name": "event_time", "type": TIMESTAMP_MILLIS},
    ]
    if dataset in DATASETS_WITH_IPS:
        fields += [{"name": "src_ip", "type": "string"}, {"name": "dst_ip", "type": "string"}]
    fields += [{"name": name, "type": "double"} for name in feature_names(dataset)]
    record = "".join(part.capitalize() for part in dataset.split("_"))
    return {"type": "record", "name": f"{record}Flow", "namespace": NAMESPACE, "fields": fields}


def label_schema() -> dict:
    """Build the delayed label schema shared by both datasets."""
    return {
        "type": "record",
        "name": "Label",
        "namespace": NAMESPACE,
        "fields": [
            {"name": "event_id", "type": "string"},
            {"name": "label", "type": "int"},
            {"name": "label_family", "type": "string"},
            {"name": "label_time", "type": TIMESTAMP_MILLIS},
        ],
    }


def prediction_schema() -> dict:
    """Build the prediction log schema."""
    return {
        "type": "record",
        "name": "Prediction",
        "namespace": NAMESPACE,
        "fields": [
            {"name": "event_id", "type": "string"},
            {"name": "dataset", "type": "string"},
            {"name": "event_time", "type": TIMESTAMP_MILLIS},
            {"name": "model_alias", "type": "string"},
            {"name": "model_version", "type": "string"},
            {"name": "score", "type": "double"},
            {"name": "decision", "type": "int"},
            {"name": "threshold", "type": "double"},
            {"name": "latency_ms", "type": "double"},
            {"name": "shadow_score", "type": ["null", "double"], "default": None},
            {"name": "shadow_version", "type": ["null", "string"], "default": None},
        ],
    }


def drift_event_schema() -> dict:
    """Build the drift event schema."""
    return {
        "type": "record",
        "name": "DriftEvent",
        "namespace": NAMESPACE,
        "fields": [
            {"name": "dataset", "type": "string"},
            {"name": "window_id", "type": "string"},
            {"name": "detected_at", "type": TIMESTAMP_MILLIS},
            {"name": "signals", "type": {"type": "map", "values": "double"}},
            {
                "name": "severity",
                "type": {"type": "enum", "name": "Severity", "symbols": ["LOW", "HIGH"]},
            },
            {"name": "policy_version", "type": "int"},
        ],
    }


def all_schemas(datasets: tuple[str, ...]) -> dict[str, dict]:
    """Return every schema keyed by its subject name."""
    schemas = {subject_name("flows", ds): flow_schema(ds) for ds in datasets}
    schemas[subject_name("labels")] = label_schema()
    schemas[subject_name("predictions")] = prediction_schema()
    schemas[subject_name("drift")] = drift_event_schema()
    return schemas


# ============= Files =============


def schema_path(subject: str) -> Path:
    """Return the committed `.avsc` path for a subject."""
    return SCHEMA_DIR / f"{subject}.avsc"


def render_schema(schema: dict) -> str:
    """Serialize a schema deterministically for committing."""
    return json.dumps(schema, indent=2) + "\n"


def load_schema(subject: str) -> dict:
    """Load a committed `.avsc` file by subject name."""
    return json.loads(schema_path(subject).read_text())
