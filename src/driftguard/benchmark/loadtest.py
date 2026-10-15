"""E3 load test: fixed-rate steps with and without a concurrent retraining run (SPEC 9, E3)."""

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from driftguard.common.logging import get_logger
from driftguard.common.mlflow_client import MlflowRegistry
from driftguard.common.settings import REPO_ROOT, get_settings

# ============= Constants =============

logger = get_logger("benchmark.e3")
CONSUMED = "driftguard_events_consumed_total"


# ============= Measurement helpers =============


def prom_query(base_url: str, promql: str) -> float | None:
    """Instant Prometheus query; None when no sample is available."""
    import httpx

    response = httpx.get(f"{base_url}/api/v1/query", params={"query": promql}, timeout=10)
    response.raise_for_status()
    result = response.json()["data"]["result"]
    if not result:
        return None
    value = float(result[0]["value"][1])
    return value if np.isfinite(value) else None


def docker_stats() -> dict[str, dict[str, float]]:
    """CPU percent and memory MiB per running container of this Compose project."""
    out = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    stats = {}
    for line in out.splitlines():
        row = json.loads(line)
        mem = row["MemUsage"].split("/")[0].strip()
        value, unit = float(mem[:-3]), mem[-3:]
        factor = {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}.get(unit, 1.0)
        stats[row["Name"]] = {
            "cpu_percent": float(row["CPUPerc"].rstrip("%")),
            "mem_mib": value * factor,
        }
    return stats


def saturated(lag_series: list[float], rate_eps: float, backlog_s: float) -> bool:
    """
    True when lag grows for the whole step.

    Lag must have a positive linear trend, end above its start, and end above `backlog_s`
    seconds of input at the step's rate.
    """
    if len(lag_series) < 3:
        return False
    lag = np.asarray(lag_series, dtype=np.float64)
    slope = np.polyfit(np.arange(len(lag)), lag, 1)[0]
    return bool(slope > 0 and lag[-1] > lag[0] and lag[-1] > backlog_s * rate_eps)


def percentile_summary(values: list[float]) -> dict[str, float | None]:
    """Max and p95 of a sampled series."""
    if not values:
        return {"max": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {"max": float(array.max()), "p95": float(np.percentile(array, 95))}


# ============= Step runner =============


@dataclass
class StepSamples:
    """Time series sampled during one step."""

    lag: list[float] = field(default_factory=list)
    latency_p50: list[float] = field(default_factory=list)
    latency_p95: list[float] = field(default_factory=list)
    latency_p99: list[float] = field(default_factory=list)
    redis_used_memory: list[float] = field(default_factory=list)
    docker: list[dict] = field(default_factory=list)


def step_quantile_query(q: float, metric: str, job: str, dataset: str, seconds: int) -> str:
    """
    PromQL for the true quantile of a latency histogram over a whole step.

    The histogram buckets are summed over the step with `increase`, so the result is the quantile
    of every observation in the step, not an average of per-window quantiles.
    """
    return (
        f"histogram_quantile({q}, sum by (le) (increase({metric}_bucket"
        f'{{job="{job}",dataset="{dataset}"}}[{int(seconds)}s])))'
    )


def latency_query(q: float, dataset: str) -> str:
    """PromQL for a scoring latency quantile from the serving router over 30 s."""
    return (
        f"histogram_quantile({q}, sum by (le) (rate(driftguard_scoring_latency_seconds_bucket"
        f'{{job="serving",dataset="{dataset}"}}[30s])))'
    )


def run_step(
    rate: float, duration_s: float, frame, dataset: str, config: dict, retrain: bool, redis_client
) -> dict[str, Any]:
    """Produce at a fixed rate for one step while sampling metrics; return the step summary."""
    from driftguard.common.metrics import scrape_metric as metric_value
    from driftguard.orchestrator.main import submit_deployment

    settings = get_settings()
    samples = StepSamples()
    consumer_url = config["consumer_metrics_url"]
    consumed_start = metric_value(consumer_url, CONSUMED, dataset=dataset)
    dlq_start = metric_value(consumer_url, "driftguard_events_dlq_total", dataset=dataset)
    produced: dict[str, Any] = {}
    producer = threading.Thread(
        target=lambda: produced.update(produce_rate(frame, dataset, rate, duration_s))
    )
    started = time.monotonic()
    producer.start()
    retrain_run, last_stats = None, 0.0
    while producer.is_alive():
        elapsed = time.monotonic() - started
        if retrain and retrain_run is None and elapsed >= float(config["retrain_at_s"]):
            retrain_run = submit_deployment(
                "retrain/retrain-on-drift",
                {
                    "dataset": dataset,
                    "trigger_id": f"e3-{int(time.time())}",
                    "window_id": "e3",
                    "window_end_ms": int(time.time() * 1000),
                },
            )
        lag = prom_query(
            settings.prometheus_url, f'sum(driftguard_consumer_lag{{dataset="{dataset}"}})'
        )
        samples.lag.append(lag or 0.0)
        for q, target in (
            (0.5, samples.latency_p50),
            (0.95, samples.latency_p95),
            (0.99, samples.latency_p99),
        ):
            value = prom_query(settings.prometheus_url, latency_query(q, dataset))
            if value is not None:
                target.append(value * 1000.0)
        samples.redis_used_memory.append(float(redis_client.info("memory")["used_memory"]))
        if time.monotonic() - last_stats >= float(config["docker_stats_interval_s"]):
            samples.docker.append(docker_stats())
            last_stats = time.monotonic()
        time.sleep(float(config["sample_interval_s"]))
    producer.join()
    elapsed = time.monotonic() - started
    consumed = metric_value(consumer_url, CONSUMED, dataset=dataset) - consumed_start
    dlq = metric_value(consumer_url, "driftguard_events_dlq_total", dataset=dataset) - dlq_start
    per_service: dict[str, dict[str, float]] = {}
    for snapshot in samples.docker:
        for name, value in snapshot.items():
            entry = per_service.setdefault(name, {"cpu_percent_max": 0.0, "mem_mib_max": 0.0})
            entry["cpu_percent_max"] = max(entry["cpu_percent_max"], value["cpu_percent"])
            entry["mem_mib_max"] = max(entry["mem_mib_max"], value["mem_mib"])
    return {
        "target_rate_eps": rate,
        "produced": produced.get("flows"),
        "achieved_produce_rate_eps": produced.get("flow_rate_eps"),
        "achieved_consumption_rate_eps": consumed / elapsed if elapsed else None,
        "lag": percentile_summary(samples.lag),
        "lag_series": samples.lag,
        "latency_ms": step_latency(settings.prometheus_url, dataset, elapsed, samples),
        "dlq_rate": dlq / consumed if consumed else None,
        "redis_peak_used_memory_bytes": max(samples.redis_used_memory, default=None),
        "per_service": per_service,
        "retraining_flow_run_id": retrain_run,
        "saturated": saturated(samples.lag, rate, float(config["saturation_backlog_s"])),
    }


def step_latency(
    prometheus_url: str, dataset: str, elapsed_s: float, samples: StepSamples
) -> dict[str, float | None]:
    """
    Step-level latency quantiles in milliseconds.

    `roundtrip_*` is measured by the consumer around each scoring request (HTTP, serialization,
    inference); `model_*` is the serving router's inference time only. `worst_30s_model_p99` is
    the largest windowed p99 seen during the step, which exposes the window containing a forced
    retraining start.
    """
    seconds = int(elapsed_s) + 5
    out: dict[str, float | None] = {}
    for prefix, metric, job in (
        ("roundtrip", "driftguard_scoring_roundtrip_seconds", "consumer"),
        ("model", "driftguard_scoring_latency_seconds", "serving"),
    ):
        for q in (0.5, 0.95, 0.99):
            value = prom_query(
                prometheus_url, step_quantile_query(q, metric, job, dataset, seconds)
            )
            out[f"{prefix}_p{int(q * 100)}"] = None if value is None else value * 1000.0
    out["worst_30s_model_p99"] = max(samples.latency_p99) if samples.latency_p99 else None
    return out


def produce_rate(frame, dataset: str, rate: float, duration_s: float) -> dict[str, Any]:
    """Produce flows from `frame` at a fixed rate for `duration_s` seconds."""
    from confluent_kafka import Producer

    from driftguard.common.kafka import producer_config
    from driftguard.data.serde import TopicSerde, registry_client
    from driftguard.producer.replay import run_replay

    settings = get_settings()
    client = registry_client(settings.schema_registry_url)
    flows, labels = TopicSerde(client, "flows", dataset), TopicSerde(client, "labels", dataset)
    producer = Producer(producer_config(settings.kafka_bootstrap_servers, "e3-producer"))

    def send(topic: str, key: str, value: bytes) -> None:
        while True:
            try:
                producer.produce(topic, key=key.encode(), value=value)
                break
            except BufferError:
                producer.poll(0.05)
        producer.poll(0)

    config = {
        "speed": {"mode": "fixed_rate", "rate_eps": rate, "burst": max(rate / 10, 1)},
        "label_delay_min": 2000,
        "label_delay_max": 10000,
        "label_coverage": 1.0,
        "duplicate_rate": 0.0,
        "seed": 42,
        "flush_every": 5000,
    }
    stats = run_replay(
        frame,
        dataset,
        config,
        send,
        flows.serialize,
        labels.serialize,
        flows.topic,
        labels.topic,
        flush=lambda: producer.flush(60),
        duration_s=duration_s,
    )
    return stats.as_dict()


# ============= Experiment =============


def clear_prefix(redis_client, prefix: str) -> int:
    """Delete every key under a prefix."""
    deleted, cursor = 0, 0
    while True:
        cursor, keys = redis_client.scan(cursor=cursor, match=f"{prefix}*", count=10_000)
        if keys:
            deleted += redis_client.delete(*keys)
        if cursor == 0:
            return deleted


def reset_step_state(redis_client, registry, dataset: str) -> None:
    """
    Return the stack to a clean, idle state before a load step.

    A forced retraining registers a challenger and switches the router to shadow, which would add
    shadow scoring load to every later step. This removes the challenger alias, restores
    `champion_only`, and clears idempotency keys (so replayed slices are scored again) and
    stream-feature window state.
    """
    from driftguard.common.mlflow_client import CHALLENGER, model_name

    registry.delete_alias(model_name(dataset), CHALLENGER)
    redis_client.set(f"dg:rollout:{dataset}", "champion_only")
    clear_prefix(redis_client, "dg:seen:")
    clear_prefix(redis_client, "dg:sf:")


def restart_consumer(mode: str) -> None:
    """Recreate the consumer container with STREAM_FEATURES_MODE set."""
    env = {**os.environ, "STREAM_FEATURES_MODE": mode}
    subprocess.run(
        ["docker", "compose", "up", "-d", "--force-recreate", "--wait", "consumer"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )


def run_e3(config: dict) -> dict[str, Any]:
    """Run every condition, stream-feature mode, and rate step."""
    from redis import Redis

    from driftguard.data.loader import load_stream_frame

    settings = get_settings()
    dataset = config["dataset"]
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    registry = MlflowRegistry(settings.mlflow_tracking_uri)
    frame = load_stream_frame(dataset)
    results: dict[str, Any] = {"partitions": config["partitions"], "runs": {}}
    for mode in config["stream_features_modes"]:
        restart_consumer(mode)
        for condition in config["conditions"]:
            steps, offset = [], 0
            for rate in config["rates_eps"]:
                reset_step_state(redis_client, registry, dataset)
                n_rows = int(rate * float(config["step_duration_s"]) * 1.2)
                if offset + n_rows > len(frame):
                    offset = 0
                window = frame.iloc[offset : offset + n_rows].reset_index(drop=True)
                offset += n_rows
                logger.info("e3_step", mode=mode, condition=condition, rate=rate)
                steps.append(
                    run_step(
                        float(rate),
                        float(config["step_duration_s"]),
                        window,
                        dataset,
                        config,
                        condition == "retraining",
                        redis_client,
                    )
                )
                time.sleep(float(config["settle_s"]))
            saturation = next((s["target_rate_eps"] for s in steps if s["saturated"]), None)
            results["runs"][f"stream_features={mode}/{condition}"] = {
                "steps": steps,
                "saturation_rate_eps": saturation,
            }
    return results
