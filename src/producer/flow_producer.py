"""Kafka producer: chronological replay of CICIDS2017 network flow records as events."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import TYPE_CHECKING

import pandas as pd

from src.config import CICIDS_COLUMN_MAP, CICIDS_DATA_DIR, CICIDS_FILES_ORDERED, settings

if TYPE_CHECKING:
    # type-only import; the native client is imported lazily at runtime
    from confluent_kafka import Producer

# ============= Configuration Constants =============

REPLAY_RATE_EPS = 500  # events per second; configurable
TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M:%S %p"
KAFKA_TOPIC = settings.kafka_topic

# fields emitted per event, in NetworkFlowEvent schema order, grouped by wire type
STRING_FIELDS = ["event_id", "flow_id", "src_ip", "dst_ip", "label"]
INT_FIELDS = ["timestamp_utc", "src_port", "dst_port", "protocol", "label_binary"]
DOUBLE_FIELDS = [
    "flow_duration",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_length_fwd_packets",
    "total_length_bwd_packets",
    "packet_length_mean",
    "packet_length_std",
    "flow_iat_mean",
    "syn_flag_count",
]
NETWORK_FLOW_FIELDS = STRING_FIELDS + INT_FIELDS + DOUBLE_FIELDS


# ============= Loading and Cleaning =============


def load_cicids_file(file_path: str, column_map: dict[str, str]) -> pd.DataFrame:
    """
    Load a single CICIDS2017 CSV file, rename columns per the field map, and
    compute binary labels.

    Parameters
    ----------
    file_path : str
        Absolute path to the CICIDS2017 CSV file.
    column_map : dict[str, str]
        Mapping from raw CSV column names to schema field names.

    Returns
    -------
    df : pd.DataFrame
        Cleaned DataFrame with renamed columns and label_binary column added.
    """
    df = pd.read_csv(file_path, low_memory=False)

    # CICIDS2017 headers carry leading and trailing whitespace
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=column_map)

    # binary label: only BENIGN is negative, every attack category is positive
    df["label_binary"] = (df["label"].astype(str).str.strip() != "BENIGN").astype(int)

    # drop rows with non-finite throughput, then zero-fill remaining gaps
    for col in ("flow_bytes_per_sec", "flow_packets_per_sec"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[df[col].notna() & ~df[col].isin([float("inf"), float("-inf")])]
    df = df.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

    # drop duplicate flow ids within the file to avoid replaying the same flow twice
    if "flow_id" in df.columns:
        df = df.drop_duplicates(subset=["flow_id"])

    return df.reset_index(drop=True)


def reindex_timestamps(
    df: pd.DataFrame,
    raw_ts_col: str,
    ts_format: str,
) -> pd.DataFrame:
    """
    Re-index raw CICIDS2017 timestamps to current wall-clock time, preserving
    relative inter-event spacing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the raw timestamp column.
    raw_ts_col : str
        Column name holding the raw timestamp strings.
    ts_format : str
        strptime format string for parsing raw timestamps.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with timestamp_utc column added as milliseconds since epoch,
        anchored to current time, sorted in ascending temporal order.
    """
    df = df.copy()

    parsed = pd.to_datetime(df[raw_ts_col], format=ts_format, errors="coerce")
    # fall back to inference for any rows the explicit format could not parse
    if parsed.isna().any():
        parsed = parsed.fillna(pd.to_datetime(df[raw_ts_col], errors="coerce"))
    df[raw_ts_col] = parsed
    df = df.dropna(subset=[raw_ts_col]).sort_values(raw_ts_col).reset_index(drop=True)

    # anchor all timestamps relative to now, preserving inter-event spacing
    origin = df[raw_ts_col].min()
    now_ms = int(time.time() * 1000)
    df["timestamp_utc"] = (
        (df[raw_ts_col] - origin).dt.total_seconds() * 1000 + now_ms
    ).astype("int64")

    return df


# ============= Event Construction =============


def build_flow_event(row: pd.Series) -> dict:
    """
    Construct a network flow event dict from a DataFrame row, injecting a
    synthetic event_id.

    Parameters
    ----------
    row : pd.Series
        Single row from the processed CICIDS2017 DataFrame.

    Returns
    -------
    event : dict
        Event dict conforming to the NetworkFlowEvent Avro schema.
    """
    event: dict = {}

    # synthetic per-record identifier used downstream for deduplication
    event["event_id"] = str(uuid.uuid4())

    for field in STRING_FIELDS:
        if field == "event_id":
            continue
        event[field] = str(row.get(field, ""))

    for field in INT_FIELDS:
        value = row.get(field, 0)
        try:
            event[field] = int(float(value))
        except (TypeError, ValueError):
            event[field] = 0

    for field in DOUBLE_FIELDS:
        value = row.get(field, 0.0)
        try:
            event[field] = float(value)
        except (TypeError, ValueError):
            event[field] = 0.0

    return event


# ============= Replay =============


def delivery_callback(err, msg) -> None:
    """Log delivery errors raised by the Kafka producer."""
    if err:
        print(f"Delivery failed: {err}")


def replay_dataset(
    producer: "Producer",  # noqa: F821 - lazy confluent_kafka type
    data_dir: str,
    file_list: list[str],
    column_map: dict[str, str],
    replay_rate_eps: int,
    avro_serializer=None,
) -> None:
    """
    Replay all CICIDS2017 files through the Kafka producer in temporal order
    at the configured rate.

    Parameters
    ----------
    producer : Producer
        Initialized Confluent Kafka producer.
    data_dir : str
        Directory containing CICIDS2017 CSV files.
    file_list : list[str]
        Ordered list of filenames to replay.
    column_map : dict[str, str]
        Field mapping from raw CSV columns to schema fields.
    replay_rate_eps : int
        Target replay rate in events per second.
    avro_serializer : AvroSerializer or None
        Registry-aware Avro serializer. When provided, events are serialized
        via the Schema Registry instead of plain JSON.
    """
    from pathlib import Path

    if avro_serializer is not None:
        from confluent_kafka.serialization import MessageField, SerializationContext

    delay_seconds = 1.0 / replay_rate_eps if replay_rate_eps > 0 else 0.0
    sent = 0

    for file_name in file_list:
        file_path = Path(data_dir) / file_name
        if not file_path.exists():
            print(f"Skipping missing file: {file_path}")
            continue

        df = load_cicids_file(str(file_path), column_map)
        df = reindex_timestamps(df, "timestamp_raw", TIMESTAMP_FORMAT)
        print(f"Replaying {len(df):,} flows from {file_name} to topic '{KAFKA_TOPIC}'")

        for _, row in df.iterrows():
            event = build_flow_event(row)
            if avro_serializer is not None:
                value = avro_serializer(
                    event, SerializationContext(KAFKA_TOPIC, MessageField.VALUE)
                )
            else:
                value = json.dumps(event).encode("utf-8")
            producer.produce(
                topic=KAFKA_TOPIC,
                value=value,
                callback=delivery_callback,
            )
            producer.poll(0)
            sent += 1

            if sent % 500 == 0:
                print(f"  Sent {sent:,} events")
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    producer.flush()
    print(f"Producer done. Total sent: {sent:,}")


def run_producer(replay_rate_eps: int = REPLAY_RATE_EPS) -> None:
    """
    Build a Kafka producer and replay all configured CICIDS2017 files.

    Parameters
    ----------
    replay_rate_eps : int
        Target replay rate in events per second.
    """
    # imported lazily so the pure replay helpers remain importable without the
    # confluent_kafka native client installed
    from confluent_kafka import Producer

    from src.schemas.registry import build_avro_serializer, register_schema

    avro_serializer = None
    try:
        register_schema()
        avro_serializer = build_avro_serializer()
        print("Schema Registry available: serializing events as Avro")
    except Exception as e:
        print(f"Schema Registry unavailable ({e}), falling back to plain JSON")

    producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
    try:
        replay_dataset(
            producer=producer,
            data_dir=str(CICIDS_DATA_DIR),
            file_list=CICIDS_FILES_ORDERED,
            column_map=CICIDS_COLUMN_MAP,
            replay_rate_eps=replay_rate_eps,
            avro_serializer=avro_serializer,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        producer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CICIDS2017 network flow Kafka producer")
    parser.add_argument(
        "--rate", type=int, default=REPLAY_RATE_EPS, help="Replay rate in events per second"
    )
    args = parser.parse_args()
    run_producer(replay_rate_eps=args.rate)
