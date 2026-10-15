"""Replay producer: emits flows in event-time order and delayed labels to Kafka (SPEC 3.4)."""

import argparse
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from driftguard.common.config import LoadedConfig, load_config
from driftguard.common.kafka import ensure_topics, producer_config
from driftguard.common.logging import configure_logging
from driftguard.common.settings import REPO_ROOT, get_settings
from driftguard.data.loader import load_stream_frame
from driftguard.data.registry import feature_names
from driftguard.data.schemas import DATASETS_WITH_IPS
from driftguard.producer.labels import DUPLICATE, FLOW, LABEL, PlanConfig, emission_plan
from driftguard.producer.rate_limit import RealtimePacer, TokenBucket

# ============= Constants =============

DEFAULT_CONFIG = REPO_ROOT / "configs" / "producer.yaml"
SPEED_MODES = ("max", "fixed_rate", "realtime_x")


# ============= Records =============


@dataclass
class ReplayStats:
    """Counts of emitted messages by kind."""

    flows: int = 0
    labels: int = 0
    duplicates: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return counts and the achieved flow rate."""
        elapsed = max(self.finished_at - self.started_at, 1e-9)
        return {
            "flows": self.flows,
            "labels": self.labels,
            "duplicates": self.duplicates,
            "elapsed_s": round(elapsed, 3),
            "flow_rate_eps": round(self.flows / elapsed, 2),
        }


def flow_record(row: dict[str, Any], dataset: str, features: list[str]) -> dict[str, Any]:
    """
    Build a flow message from one adapter row; the label is never included.

    Parameters
    ----------
    row : dict
        Adapter row with identifiers, event time, optional IPs, and features.
    dataset : str
        Dataset name.
    features : list of str
        Ordered feature names.

    Returns
    -------
    record : dict
        Value matching `flows-<dataset>-value`.
    """
    record = {
        "event_id": row["event_id"],
        "dataset": dataset,
        "source_file": row["source_file"],
        "row_index": int(row["row_index"]),
        "event_time": int(row["event_time"]),
    }
    if dataset in DATASETS_WITH_IPS:
        record["src_ip"] = str(row["src_ip"])
        record["dst_ip"] = str(row["dst_ip"])
    for name in features:
        record[name] = float(row[name])
    return record


def label_record(row: dict[str, Any], label_time_ms: int) -> dict[str, Any]:
    """Build a delayed label message; `label_time` is the event time at release."""
    return {
        "event_id": row["event_id"],
        "label": int(row["label"]),
        "label_family": str(row["label_family"]),
        "label_time": int(label_time_ms),
    }


# ============= Plan execution =============


def plan_config_from(config: LoadedConfig | dict) -> PlanConfig:
    """Build the emission plan parameters, applying the selected profile's overrides."""
    data = dict(config.data if isinstance(config, LoadedConfig) else config)
    profile = (data.get("profiles") or {}).get(data.get("profile") or "", {}) or {}
    data.update(profile)
    return PlanConfig(
        label_delay_min=int(data["label_delay_min"]),
        label_delay_max=int(data["label_delay_max"]),
        label_coverage=float(data["label_coverage"]),
        duplicate_rate=float(data["duplicate_rate"]),
        duplicate_delay_max=int(data.get("duplicate_delay_max", 100)),
        seed=int(data["seed"]),
    )


def iter_messages(
    frame: pd.DataFrame, dataset: str, plan: PlanConfig
) -> Iterator[tuple[str, str, dict[str, Any], int]]:
    """
    Yield messages in emission order.

    Yields
    ------
    message : tuple
        (kind, event_id, record, event_time_ms of the most recent flow), where kind is
        `flow`, `label`, or `duplicate`.
    """
    features = feature_names(dataset)
    # per-column arrays keep native dtypes; an object matrix of the full frame would not fit in RAM
    column_arrays = {name: frame[name].to_numpy() for name in frame.columns}
    event_time = frame["event_time"].to_numpy(dtype=np.int64)
    clock_ms = int(event_time[0]) if len(event_time) else 0

    def row_at(index: int) -> dict[str, Any]:
        return {name: array[index] for name, array in column_arrays.items()}

    for kind, index in emission_plan(len(frame), plan):
        row = row_at(index)
        if kind == FLOW:
            clock_ms = int(event_time[index])
            yield FLOW, row["event_id"], flow_record(row, dataset, features), clock_ms
        elif kind == DUPLICATE:
            yield DUPLICATE, row["event_id"], flow_record(row, dataset, features), clock_ms
        else:
            yield LABEL, row["event_id"], label_record(row, clock_ms), clock_ms


def make_pacer(speed: dict[str, Any]) -> Callable[[int], None]:
    """Return a per-flow pacing callable for the configured speed mode."""
    mode = speed.get("mode", "max")
    if mode not in SPEED_MODES:
        raise ValueError(f"speed.mode must be one of {SPEED_MODES}")
    if mode == "fixed_rate":
        bucket = TokenBucket(float(speed["rate_eps"]), float(speed.get("burst", speed["rate_eps"])))
        return lambda _event_time_ms: bucket.acquire()
    if mode == "realtime_x":
        pacer = RealtimePacer(float(speed["realtime_factor"]))
        return pacer.wait_until
    return lambda _event_time_ms: None


def run_replay(
    frame: pd.DataFrame,
    dataset: str,
    config: LoadedConfig | dict,
    send: Callable[[str, str, bytes], None],
    serialize_flow: Callable[[dict[str, Any]], bytes],
    serialize_label: Callable[[dict[str, Any]], bytes],
    flow_topic: str,
    label_topic: str,
    flush: Callable[[], None] = lambda: None,
    duration_s: float | None = None,
) -> ReplayStats:
    """
    Emit the replay through injected transport callables.

    Parameters
    ----------
    frame : pd.DataFrame
        Stream-ordered adapter frame.
    dataset : str
        Dataset name.
    config : LoadedConfig or dict
        Producer config (speed, plan, flush cadence).
    send : callable
        `send(topic, key, value_bytes)`.
    serialize_flow, serialize_label : callable
        Record encoders.
    flow_topic, label_topic : str
        Destination topics.
    flush : callable
        Called every `flush_every` flows and at the end.
    duration_s : float or None
        Stop emitting new flows after this many seconds (load tests).

    Returns
    -------
    stats : ReplayStats
        Message counts and the achieved flow rate.
    """
    data = config.data if isinstance(config, LoadedConfig) else config
    pace = make_pacer(data.get("speed", {}))
    flush_every = int(data.get("flush_every", 5000))
    stats = ReplayStats(started_at=time.monotonic())
    deadline = stats.started_at + duration_s if duration_s else None
    for kind, event_id, record, clock_ms in iter_messages(frame, dataset, plan_config_from(data)):
        if kind == FLOW:
            if deadline is not None and time.monotonic() >= deadline:
                break
            pace(clock_ms)
            send(flow_topic, event_id, serialize_flow(record))
            stats.flows += 1
            if stats.flows % flush_every == 0:
                flush()
        elif kind == DUPLICATE:
            send(flow_topic, event_id, serialize_flow(record))
            stats.duplicates += 1
        else:
            send(label_topic, event_id, serialize_label(record))
            stats.labels += 1
    flush()
    stats.finished_at = time.monotonic()
    return stats


def select_slice(frame: pd.DataFrame, start_index: int, limit: int | None) -> pd.DataFrame:
    """Return the replay slice `[start_index, start_index + limit)`."""
    end = None if limit is None else start_index + int(limit)
    return frame.iloc[start_index:end].reset_index(drop=True)


# ============= CLI =============


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse producer command-line overrides."""
    parser = argparse.ArgumentParser(description="DriftGuard replay producer")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset")
    parser.add_argument("--source", choices=["dataset", "synthetic"])
    parser.add_argument("--mode", choices=list(SPEED_MODES))
    parser.add_argument("--rate", type=float)
    parser.add_argument("--realtime-factor", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--profile")
    parser.add_argument("--duplicate-rate", type=float)
    parser.add_argument("--n-events", type=int, help="synthetic source size")
    return parser.parse_args(argv)


def apply_overrides(config: LoadedConfig, args: argparse.Namespace) -> dict[str, Any]:
    """Merge command-line overrides into a copy of the config data."""
    data = {**config.data, "speed": dict(config["speed"])}
    data["synthetic"] = dict(config.get("synthetic", {}) or {})
    for key, value in [
        ("dataset", args.dataset),
        ("source", args.source),
        ("limit", args.limit),
        ("start_index", args.start_index),
        ("duration_s", args.duration_s),
        ("profile", args.profile),
        ("duplicate_rate", args.duplicate_rate),
    ]:
        if value is not None:
            data[key] = value
    if args.mode:
        data["speed"]["mode"] = args.mode
    if args.rate:
        data["speed"]["rate_eps"] = args.rate
        data["speed"]["burst"] = max(1.0, args.rate)
    if args.realtime_factor:
        data["speed"]["realtime_factor"] = args.realtime_factor
    if args.n_events:
        data["synthetic"]["n_events"] = args.n_events
    return data


def main(argv: list[str] | None = None) -> None:
    """Run the replay against Kafka with Avro serialization."""
    from confluent_kafka import Producer

    from driftguard.data.serde import TopicSerde, registry_client

    args = parse_args(argv)
    settings = get_settings()
    logger = configure_logging("producer", settings.log_level)
    config = load_config(args.config)
    data = apply_overrides(config, args)
    dataset = data["dataset"]

    ensure_topics(settings.kafka_bootstrap_servers, (dataset,))
    client = registry_client(settings.schema_registry_url)
    flow_serde = TopicSerde(client, "flows", dataset)
    label_serde = TopicSerde(client, "labels", dataset)
    producer = Producer(producer_config(settings.kafka_bootstrap_servers, f"producer-{dataset}"))

    def send(topic: str, key: str, value: bytes) -> None:
        while True:
            try:
                producer.produce(topic, key=key.encode(), value=value)
                break
            except BufferError:
                producer.poll(0.1)
        producer.poll(0)

    frame = load_stream_frame(dataset, data["source"], config)
    frame = select_slice(frame, int(data.get("start_index", 0)), data.get("limit"))
    logger.info("replay_start", dataset=dataset, events=len(frame), speed=data["speed"])
    stats = run_replay(
        frame,
        dataset,
        data,
        send,
        flow_serde.serialize,
        label_serde.serialize,
        flow_serde.topic,
        label_serde.topic,
        flush=lambda: producer.flush(30),
        duration_s=data.get("duration_s"),
    )
    logger.info("replay_done", dataset=dataset, **stats.as_dict())


if __name__ == "__main__":
    main()
