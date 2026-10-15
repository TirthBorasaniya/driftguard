"""E4 end-to-end adaptation on the live stack: static versus adaptive (SPEC 9, E4)."""

import os
import subprocess
import threading
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from driftguard.common.logging import get_logger
from driftguard.common.mlflow_client import (
    CHALLENGER,
    CHAMPION,
    PREVIOUS,
    MlflowRegistry,
    model_name,
)
from driftguard.common.promotions import read_promotions
from driftguard.common.settings import REPO_ROOT, get_settings

# ============= Constants =============

logger = get_logger("benchmark.e4")
PAUSE_KEY = "dg:retrain:paused:{dataset}"


# ============= Metrics from collected streams =============


def pr_auc_or_none(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """PR-AUC, or None when a slice has no positives."""
    return float(average_precision_score(labels, scores)) if labels.sum() else None


def per_period_pr_auc(frame: pd.DataFrame, period: pd.Series) -> dict[str, dict[str, Any]]:
    """PR-AUC of the served score per period (year or month), with counts."""
    out = {}
    for key, part in frame.groupby(period, sort=True):
        out[str(key)] = {
            "pr_auc": pr_auc_or_none(part["label"].to_numpy(), part["score"].to_numpy()),
            "n": len(part),
            "n_positive": int(part["label"].sum()),
        }
    return out


def rolling_pr_auc(frame: pd.DataFrame, window: int) -> list[dict[str, Any]]:
    """PR-AUC over consecutive blocks of `window` events in event-time order."""
    ordered = frame.sort_values(["event_time", "event_id"], kind="mergesort")
    rows = []
    for start in range(0, len(ordered), window):
        part = ordered.iloc[start : start + window]
        rows.append(
            {
                "start_event_time": int(part["event_time"].iloc[0]),
                "end_event_time": int(part["event_time"].iloc[-1]),
                "pr_auc": pr_auc_or_none(part["label"].to_numpy(), part["score"].to_numpy()),
                "n": len(part),
            }
        )
    return rows


def summarize_lifecycle(promotions: list[dict], drift_events: list[dict]) -> dict[str, Any]:
    """Counts of drift events and gate outcomes during a run."""
    actions = [p["action"] for p in promotions]
    return {
        "drift_events": len(drift_events),
        "drift_events_high": sum(e["severity"] == "HIGH" for e in drift_events),
        "promotions": actions.count("promote"),
        "promoted_to_canary": actions.count("promote_to_canary"),
        "declines": actions.count("decline"),
        "rollbacks": actions.count("canary_rolled_back") + actions.count("rollback"),
    }


def wall_to_event_time(ts_iso: str, progress: list[tuple[int, int]]) -> int | None:
    """Event time of the newest prediction seen at a wall-clock timestamp (ISO 8601, UTC)."""
    wall_ms = int(
        datetime.strptime(ts_iso[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC).timestamp() * 1000
    )
    seen = [event for wall, event in progress if wall <= wall_ms]
    return seen[-1] if seen else None


# ============= Stack control =============


def compose(*args: str, dataset: str) -> None:
    """Run docker compose with DATASET set for the app services."""
    subprocess.run(
        ["docker", "compose", *args],
        check=True,
        cwd=REPO_ROOT,
        env={**os.environ, "DATASET": dataset},
    )


def reset_for_run(dataset: str, initial_version: str, redis_client, registry) -> None:
    """Restore the initial champion, clear challenger state, idempotency keys, and pause flags."""
    name = model_name(dataset)
    registry.set_alias(name, CHAMPION, initial_version)
    registry.delete_alias(name, CHALLENGER)
    registry.delete_alias(name, PREVIOUS)
    redis_client.set(f"dg:rollout:{dataset}", "champion_only")
    redis_client.delete(PAUSE_KEY.format(dataset=dataset))
    cursor = 0
    while True:
        cursor, keys = redis_client.scan(cursor=cursor, match="dg:seen:*", count=10_000)
        if keys:
            redis_client.delete(*keys)
        if cursor == 0:
            break


class StreamCollector:
    """Collects predictions, labels, and drift events produced after a start timestamp."""

    def __init__(self, dataset: str, start_ms: int):
        from confluent_kafka import Consumer

        from driftguard.common.kafka import consumer_config
        from driftguard.data.schemas import topic_name
        from driftguard.data.serde import TopicSerde, registry_client

        settings = get_settings()
        client = registry_client(settings.schema_registry_url)
        self.serdes = {
            topic_name("predictions"): TopicSerde(client, "predictions"),
            topic_name("labels", dataset): TopicSerde(client, "labels", dataset),
            topic_name("drift"): TopicSerde(client, "drift"),
        }
        self.consumer = Consumer(
            consumer_config(settings.kafka_bootstrap_servers, f"e4-{dataset}-{start_ms}", "e4")
        )
        self.dataset, self.start_ms = dataset, start_ms
        self.records: dict[str, list[dict]] = {t: [] for t in self.serdes}
        self.last_message = time.monotonic()
        # (wall-clock ms, newest prediction event time) pairs, to place gate decisions in event time
        self.progress: list[tuple[int, int]] = []
        self._newest_event_time = 0
        self._stop = threading.Event()

    def run(self) -> None:
        """Seek every partition to `start_ms`, then consume until stopped."""
        from confluent_kafka import TopicPartition

        metadata = self.consumer.list_topics(timeout=10)
        partitions = [
            TopicPartition(t, p, self.start_ms)
            for t in self.serdes
            for p in metadata.topics[t].partitions
        ]
        self.consumer.assign(self.consumer.offsets_for_times(partitions, timeout=10))
        while not self._stop.is_set():
            for message in self.consumer.consume(num_messages=5000, timeout=1.0):
                if message.error() or message.offset() < 0:
                    continue
                record = self.serdes[message.topic()].deserialize(message.value())
                if record.get("dataset", self.dataset) != self.dataset:
                    continue
                self.records[message.topic()].append(self._slim(record))
                self.last_message = time.monotonic()
                if "score" in record:
                    self._newest_event_time = max(self._newest_event_time, record["event_time"])
            if self._newest_event_time:
                self.progress.append((int(time.time() * 1000), self._newest_event_time))
        self.consumer.close()

    @staticmethod
    def _slim(record: dict) -> dict:
        """Keep only the fields E4 needs; millions of full records would not fit in memory."""
        if "score" in record:
            return {k: record[k] for k in ("event_id", "event_time", "score", "model_alias")}
        if "label" in record:
            return {"event_id": record["event_id"], "label": record["label"]}
        return record

    def stop(self) -> None:
        """Stop consuming."""
        self._stop.set()


def run_mode(
    dataset: str, mode: str, frame: pd.DataFrame, rate_eps: float, initial: str, config: dict
) -> dict[str, Any]:
    """Replay the stream once in `static` or `adaptive` mode and collect everything."""
    from redis import Redis

    from driftguard.benchmark.loadtest import produce_rate
    from driftguard.data.schemas import topic_name

    settings = get_settings()
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    registry = MlflowRegistry(settings.mlflow_tracking_uri)
    reset_for_run(dataset, initial, redis_client, registry)
    if mode == "static":
        redis_client.set(PAUSE_KEY.format(dataset=dataset), "operator:e4-static")
    started_ms = int(time.time() * 1000)
    collector = StreamCollector(dataset, started_ms)
    thread = threading.Thread(target=collector.run, daemon=True)
    thread.start()
    duration = len(frame) / rate_eps + 60
    produced = produce_rate(frame, dataset, rate_eps, duration)
    deadline = time.monotonic() + float(config["drain_timeout_s"])
    preds_topic = topic_name("predictions")
    while time.monotonic() < deadline:
        done = len(collector.records[preds_topic]) >= produced["flows"]
        idle = time.monotonic() - collector.last_message > float(config["drain_idle_s"])
        if done and idle:
            break
        time.sleep(10)
    collector.stop()
    thread.join(timeout=30)
    started_iso = datetime.fromtimestamp(started_ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    promotions = [
        p
        for p in read_promotions(settings.promotions_log_path)
        if p["dataset"] == dataset and p["ts"] >= started_iso
    ]
    predictions = pd.DataFrame(collector.records[preds_topic]).drop_duplicates("event_id")
    labels = pd.DataFrame(collector.records[topic_name("labels", dataset)]).drop_duplicates(
        "event_id"
    )
    joined = predictions.merge(labels[["event_id", "label"]], on="event_id")
    drift = collector.records[topic_name("drift")]
    if mode == "static":
        redis_client.delete(PAUSE_KEY.format(dataset=dataset))
    return {
        "produced": produced,
        "predictions": len(predictions),
        "labeled": len(joined),
        "joined": joined,
        "drift_events": drift,
        "promotions": promotions,
        "progress": collector.progress,
        "lifecycle": summarize_lifecycle(promotions, drift),
    }


def run_dataset(dataset: str, config: dict) -> dict[str, Any]:
    """Bring the stack up for one dataset, bootstrap, and run its modes."""
    from driftguard.data.loader import load_stream_frame

    compose("up", "-d", "--wait", dataset=dataset)
    compose(
        "run",
        "--rm",
        "bootstrap",
        "python",
        "scripts/bootstrap.py",
        "--dataset",
        dataset,
        dataset=dataset,
    )
    registry = MlflowRegistry(get_settings().mlflow_tracking_uri)
    initial = registry.get_alias(model_name(dataset), CHAMPION).version
    frame = load_stream_frame(dataset)
    out: dict[str, Any] = {"initial_champion": initial, "stream_rows": len(frame), "modes": {}}
    for mode in config[dataset]["modes"]:
        logger.info("e4_mode", dataset=dataset, mode=mode)
        result = run_mode(dataset, mode, frame, float(config[dataset]["rate_eps"]), initial, config)
        joined = result.pop("joined")
        progress = result.pop("progress")
        if dataset == "kyoto":
            years = pd.to_datetime(joined["event_time"], unit="ms", utc=True).dt.year
            result["pr_auc_per_year"] = per_period_pr_auc(joined, years)
        else:
            result["rolling_pr_auc"] = rolling_pr_auc(
                joined, int(config[dataset]["rolling_window_events"])
            )
            result["timeline"] = {
                "drift_events": [
                    {k: e[k] for k in ("window_id", "detected_at", "severity")}
                    | {"window_end_ms": e["signals"].get("window_end_ms")}
                    for e in result["drift_events"]
                ],
                "gate_decisions": [
                    p | {"event_time_ms": wall_to_event_time(p["ts"], progress)}
                    for p in result["promotions"]
                ],
            }
        result["drift_events"] = len(result["drift_events"])
        out["modes"][mode] = result
    return out
