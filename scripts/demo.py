"""Demo: one injected drift through detection, retraining, shadow, gate decision, and rollback."""

import argparse
import threading
import time
import uuid
from datetime import UTC, datetime

import httpx
import pandas as pd

from driftguard.benchmark.injector import inject
from driftguard.benchmark.loadtest import produce_rate
from driftguard.common.config import load_config
from driftguard.common.metrics import scrape_metric
from driftguard.common.mlflow_client import CHALLENGER, MlflowRegistry, model_name
from driftguard.common.promotions import read_promotions
from driftguard.common.settings import REPO_ROOT, get_settings
from driftguard.data.loader import load_stream_frame
from driftguard.data.schemas import topic_name

# ============= Constants =============

DATASET = "cicids2017"
SHIFTED_FEATURES = ["flow_duration", "flow_bytes_per_sec", "flow_iat_mean"]
SHIFT_PARAMS = {"features": SHIFTED_FEATURES, "factor": 2.0, "noise_std": 0.1}
SERVING_METRICS = "http://localhost:8001/metrics"
SAFE_PREFIX = "dg:seen:"


# ============= Narration =============


def say(stage: str, message: str) -> None:
    """Print one timestamped line of the narrative."""
    print(f"[{datetime.now(UTC):%H:%M:%S}] {stage:<10} {message}", flush=True)


# ============= Stream =============


def build_stream(
    source: str, window: int, onset_window: int, total_events: int, slice_events: int
) -> tuple[pd.DataFrame, int]:
    """
    Tile a stationary slice into a long stream and inject a covariate shift at a known onset.

    Each repetition gets fresh event ids and event times advanced by the slice's span, so window
    state advances and idempotency keys do not collide. Returns the stream and the onset index.
    """
    frame = load_stream_frame(DATASET, source)
    if source == "dataset":
        frame = frame[frame["day"] == "tuesday"]
    base = frame.head(slice_events).reset_index(drop=True)
    span = int(base["event_time"].max() - base["event_time"].min()) + 1000
    tag = uuid.uuid4().hex[:6]
    reps = -(-total_events // len(base))
    parts = []
    for k in range(reps):
        part = base.copy()
        part["event_id"] = [f"{eid}#{tag}{k}" for eid in part["event_id"]]
        part["event_time"] = part["event_time"] + k * span
        parts.append(part)
    stream = pd.concat(parts, ignore_index=True).head(total_events)
    onset = onset_window * window
    shifted, _ = inject(stream, "covariate", onset, SHIFT_PARAMS, seed=3, reference=base)
    return shifted, onset


def clear_idempotency(redis_client) -> int:
    """Delete idempotency keys so a re-run of the demo is scored again."""
    deleted, cursor = 0, 0
    while True:
        cursor, keys = redis_client.scan(cursor=cursor, match=f"{SAFE_PREFIX}*", count=10_000)
        if keys:
            deleted += redis_client.delete(*keys)
        if cursor == 0:
            return deleted


class DriftEventWatcher:
    """Reads `drift.events` newer than the start of the demo, without blocking."""

    def __init__(self, start_ms: int):
        from confluent_kafka import Consumer

        from driftguard.common.kafka import consumer_config
        from driftguard.data.serde import TopicSerde, registry_client

        settings = get_settings()
        self.serde = TopicSerde(registry_client(settings.schema_registry_url), "drift")
        self.consumer = Consumer(
            consumer_config(settings.kafka_bootstrap_servers, f"demo-{uuid.uuid4().hex}", "demo")
        )
        self.consumer.subscribe([topic_name("drift")])
        self.start_ms = start_ms

    def poll(self) -> list[dict]:
        """Return drift events detected since the demo started."""
        events = []
        for message in self.consumer.consume(num_messages=50, timeout=0.5):
            if message.error():
                continue
            event = self.serde.deserialize(message.value())
            if event["detected_at"] >= self.start_ms:
                events.append(event)
        return events


# ============= Run =============


def main() -> None:
    """Run the demo against the bootstrapped stack and narrate each stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["dataset", "synthetic"], default="dataset")
    parser.add_argument("--rate", type=float, default=1000.0)
    parser.add_argument("--events", type=int, default=600_000)
    parser.add_argument("--onset-window", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=2400.0)
    args = parser.parse_args()

    from redis import Redis

    settings = get_settings()
    policy = load_config(REPO_ROOT / "configs" / "drift_policy.yaml").data
    window = int(policy["window_size"])
    registry = MlflowRegistry(settings.mlflow_tracking_uri)
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    name = model_name(DATASET)

    champion = registry.get_alias(name, "champion")
    if champion is None:
        raise SystemExit("no champion registered; run `make bootstrap` first")
    say(
        "start",
        f"champion is version {champion.version}; {clear_idempotency(redis_client)} "
        "old idempotency keys cleared",
    )
    stream, onset = build_stream(
        args.source,
        window,
        args.onset_window,
        args.events,
        slice_events=100_000 if args.source == "dataset" else 10_000,
    )
    say(
        "stream",
        f"{len(stream):,} events at {args.rate:g}/s; covariate shift (x2 on "
        f"{', '.join(SHIFTED_FEATURES)}) starts at event {onset:,} (window "
        f"{args.onset_window})",
    )

    start_ms = int(time.time() * 1000)
    start_iso = datetime.fromtimestamp(start_ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
    producer = threading.Thread(
        target=lambda: produce_rate(stream, DATASET, args.rate, len(stream) / args.rate + 60),
        daemon=True,
    )
    producer.start()
    watcher = DriftEventWatcher(start_ms)

    announced: set[str] = set()
    seen_decisions = 0
    promoted = False
    deadline = time.monotonic() + args.timeout_s
    while time.monotonic() < deadline:
        for event in watcher.poll():
            key = f"drift:{event['window_id']}"
            if key not in announced:
                announced.add(key)
                say(
                    "drift",
                    f"event at {event['window_id']}, severity {event['severity']}, "
                    f"psi_max {event['signals'].get('psi_max', float('nan')):.3f}",
                )
        challenger = registry.get_alias(name, CHALLENGER)
        if challenger and "challenger" not in announced:
            announced.add("challenger")
            say(
                "retrain",
                f"challenger version {challenger.version} registered; rollout mode "
                f"{redis_client.get(f'dg:rollout:{DATASET}')} (shadow evaluation)",
            )
        decisions = [
            p
            for p in read_promotions(settings.promotions_log_path)
            if p["dataset"] == DATASET and p["ts"] >= start_iso
        ]
        for entry in decisions[seen_decisions:]:
            seen_decisions += 1
            if entry["action"] in ("promote_to_canary", "decline"):
                say(
                    "gate",
                    f"{entry['action']}: delta PR-AUC {entry['delta_pr_auc']:+.4f}, "
                    f"95% CI [{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}], "
                    f"{entry['n_events']:,} labeled shadow events",
                )
            else:
                say("canary", f"{entry['action']}: champion {entry.get('champion')}")
            promoted = promoted or entry["action"] == "promote"
        if promoted or any(p["action"] in ("decline", "canary_rolled_back") for p in decisions):
            break
        time.sleep(5)
    else:
        raise SystemExit("timed out before a gate decision was recorded")

    if not promoted:
        say(
            "rollback",
            "no promotion happened, so there is no previous champion to restore; "
            "re-run the demo or lower promotion_margin in a copy of configs/rollout.yaml",
        )
        return
    promoted_to = registry.get_alias(name, "champion").version
    before = scrape_metric(
        SERVING_METRICS, "driftguard_model_info", alias="champion", version=champion.version
    )
    say("rollback", f"champion is now version {promoted_to}; rolling back")
    started = time.monotonic()
    response = httpx.post(
        f"{settings.serving_url}/admin/rollback",
        json={"dataset": DATASET, "reason": "demo"},
        headers={"Authorization": f"Bearer {settings.admin_token.get_secret_value()}"},
        timeout=30,
    )
    response.raise_for_status()
    while time.monotonic() - started < 30:
        if (
            scrape_metric(
                SERVING_METRICS, "driftguard_model_info", alias="champion", version=champion.version
            )
            == 1.0
            and before != 1.0
        ):
            say(
                "rollback",
                f"serving restored version {champion.version} in "
                f"{time.monotonic() - started:.1f} s",
            )
            return
        time.sleep(0.5)
    say("rollback", "alias moved but serving did not report the restored version within 30 s")


if __name__ == "__main__":
    main()
