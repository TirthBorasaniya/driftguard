"""Rollout controller: shadow gate, canary guardrails, promotion, and automatic rollback."""

import signal
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from driftguard.common.config import load_config
from driftguard.common.logging import get_logger
from driftguard.common.metrics import SHADOW_AGREEMENT
from driftguard.common.mlflow_client import CHALLENGER, model_name
from driftguard.common.settings import get_settings
from driftguard.serving import admin
from driftguard.serving.router import MODE_KEY
from driftguard.training.gate import shadow_gate

# ============= Constants =============

logger = get_logger("orchestrator.rollout")
MAX_PENDING = 500_000


# ============= Guardrails =============


def guardrail_queries(dataset: str, window: str) -> dict[str, str]:
    """PromQL for the canary guardrails, evaluated on the serving router's metrics."""
    sel = f'job="serving",dataset="{dataset}"'
    rate = "sum(rate(driftguard_{m}{{{s}}}[{w}]))"

    def positive_rate(alias: str) -> str:
        s = f'{sel},alias="{alias}"'
        return (
            f"{rate.format(m='positive_decisions_total', s=s, w=window)} / "
            f"{rate.format(m='scored_events_total', s=s, w=window)}"
        )

    return {
        "p99_latency_s": (
            "histogram_quantile(0.99, sum by (le) "
            f"(rate(driftguard_scoring_latency_seconds_bucket{{{sel}}}[{window}])))"
        ),
        "error_rate": (
            f"{rate.format(m='scoring_errors_total', s=sel, w=window)} / "
            f"{rate.format(m='scoring_requests_total', s=sel, w=window)}"
        ),
        "positive_rate_ratio": f"({positive_rate('challenger')}) / ({positive_rate('champion')})",
    }


def evaluate_guardrails(values: dict[str, float | None], guardrails: dict[str, Any]) -> list[str]:
    """
    Return the names of breached guardrails.

    Missing values (no traffic yet in the window) are not breaches; NaN error rates with zero
    requests are treated the same way.
    """
    breaches = []
    p99 = values.get("p99_latency_s")
    if p99 is not None and np.isfinite(p99) and p99 * 1000.0 >= float(guardrails["p99_latency_ms"]):
        breaches.append("p99_latency")
    err = values.get("error_rate")
    if err is not None and np.isfinite(err) and err >= float(guardrails["error_rate"]):
        breaches.append("error_rate")
    ratio = values.get("positive_rate_ratio")
    low, high = guardrails["positive_rate_ratio"]
    if ratio is not None and np.isfinite(ratio) and not float(low) <= ratio <= float(high):
        breaches.append("positive_rate_ratio")
    return breaches


# ============= Controller =============


@dataclass
class ShadowPair:
    """Champion and challenger scores of one shadow-scored event."""

    champion: float
    challenger: float
    champion_decision: int
    challenger_version: str


@dataclass
class RolloutController:
    """
    Transport-free rollout state machine.

    Parameters
    ----------
    dataset : str
        Dataset.
    config : dict
        `configs/rollout.yaml` content.
    registry : Registry
        Model registry (alias moves).
    redis_client : redis.Redis
        Rollout mode store.
    log_path : Path
        `results/promotions.jsonl`.
    query : callable
        `query(promql) -> float | None` against Prometheus.
    clock : callable
        Monotonic clock.
    """

    dataset: str
    config: dict[str, Any]
    registry: Any
    redis: Any
    log_path: Any
    query: Callable[[str], float | None]
    clock: Callable[[], float] = time.monotonic
    pending: OrderedDict = field(default_factory=OrderedDict)
    labels: OrderedDict = field(default_factory=OrderedDict)
    labeled: list[tuple[int, float, float, int]] = field(default_factory=list)
    challenger_version: str | None = None
    canary_events: int = 0
    last_guardrail_check: float = float("-inf")
    decisions: list[dict[str, Any]] = field(default_factory=list)

    def mode(self) -> str:
        """Current rollout mode."""
        return self.redis.get(MODE_KEY.format(dataset=self.dataset)) or "champion_only"

    def _challenger(self) -> str | None:
        ref = self.registry.get_alias(model_name(self.dataset), CHALLENGER)
        return ref.version if ref else None

    def _sync_challenger(self) -> None:
        version = self._challenger()
        if version != self.challenger_version:
            self.challenger_version = version
            self.pending.clear()
            self.labels.clear()
            self.labeled = []
            self.canary_events = 0

    def on_predictions(self, records: list[dict[str, Any]]) -> None:
        """Collect shadow pairs (shadow mode) and count challenger-served events (canary)."""
        self._sync_challenger()
        mode = self.mode()
        for r in records:
            if mode == "canary" and r["model_alias"] == CHALLENGER:
                self.canary_events += 1
            if r.get("shadow_score") is None or r.get("shadow_version") != self.challenger_version:
                continue
            pair = ShadowPair(
                float(r["score"]),
                float(r["shadow_score"]),
                int(r["decision"]),
                str(r["shadow_version"]),
            )
            label = self.labels.pop(r["event_id"], None)
            if label is None:
                self.pending[r["event_id"]] = pair
                if len(self.pending) > MAX_PENDING:
                    self.pending.popitem(last=False)
            else:
                self._add_labeled(label, pair)

    def on_labels(self, records: list[dict[str, Any]]) -> None:
        """Join labels with pending shadow pairs."""
        for r in records:
            pair = self.pending.pop(r["event_id"], None)
            if pair is None:
                self.labels[r["event_id"]] = int(r["label"])
                if len(self.labels) > MAX_PENDING:
                    self.labels.popitem(last=False)
            else:
                self._add_labeled(int(r["label"]), pair)

    def _add_labeled(self, label: int, pair: ShadowPair) -> None:
        self.labeled.append((label, pair.champion, pair.challenger, pair.champion_decision))

    def step(self) -> dict[str, Any] | None:
        """Advance the state machine; returns a decision record when an action was taken."""
        self._sync_challenger()
        mode = self.mode()
        if mode == "shadow" and self.challenger_version is not None:
            return self._maybe_gate()
        if mode == "canary" and self.challenger_version is not None:
            return self._maybe_finish_canary()
        return None

    def _maybe_gate(self) -> dict[str, Any] | None:
        if len(self.labeled) < int(self.config["shadow_min_labeled"]):
            return None
        rows = np.asarray(self.labeled[: int(self.config["shadow_min_labeled"])])
        y, champ, chal = rows[:, 0].astype(int), rows[:, 1], rows[:, 2]
        champ_threshold = self.registry.get_alias(model_name(self.dataset), "champion").threshold
        chal_threshold = self.registry.get_alias(model_name(self.dataset), CHALLENGER).threshold
        agreement = float(np.mean((champ >= champ_threshold) == (chal >= chal_threshold)))
        SHADOW_AGREEMENT.labels(dataset=self.dataset).set(agreement)
        if y.sum() == 0:
            logger.warning("shadow_gate_waiting_for_positives", labeled=len(y))
            return None
        gate = shadow_gate(
            y,
            champ,
            chal,
            float(self.config["promotion_margin"]),
            int(self.config["bootstrap_resamples"]),
            int(self.config["bootstrap_seed"]),
        )
        evidence = {**gate.as_dict(), "shadow_agreement_ratio": agreement}
        if gate.decision == "promote":
            admin.set_mode(
                self.redis, self.dataset, "canary", float(self.config["canary_fraction"])
            )
            self.canary_events = 0
            record = {
                "action": "promote_to_canary",
                "challenger": self.challenger_version,
                **evidence,
            }
            from driftguard.common.promotions import append_promotion

            append_promotion(self.log_path, {"dataset": self.dataset, **record})
        else:
            record = admin.decline(
                self.registry, self.redis, self.dataset, self.log_path, "decline", evidence
            )
        self.decisions.append(record)
        logger.info(
            "shadow_gate_decided",
            decision=gate.decision,
            delta=round(gate.delta_pr_auc, 4),
            ci_low=round(gate.ci_low, 4),
        )
        return record

    def check_guardrails(self) -> tuple[list[str], dict[str, float | None]]:
        """Query Prometheus and return breached guardrails with the values seen."""
        queries = guardrail_queries(self.dataset, str(self.config["guardrail_window"]))
        values = {name: self.query(promql) for name, promql in queries.items()}
        return evaluate_guardrails(values, self.config["guardrails"]), values

    def _maybe_finish_canary(self) -> dict[str, Any] | None:
        now = self.clock()
        if now - self.last_guardrail_check >= float(self.config["guardrail_check_s"]):
            self.last_guardrail_check = now
            breaches, values = self.check_guardrails()
            if breaches:
                record = admin.decline(
                    self.registry,
                    self.redis,
                    self.dataset,
                    self.log_path,
                    "canary_rolled_back",
                    {
                        "breaches": breaches,
                        "guardrail_values": values,
                        "canary_events": self.canary_events,
                    },
                )
                self.decisions.append(record)
                logger.warning("canary_rolled_back", breaches=breaches)
                return record
        if self.canary_events >= int(self.config["canary_min_events"]):
            breaches, values = self.check_guardrails()
            if breaches:
                return None
            record = admin.promote(
                self.registry,
                self.redis,
                self.dataset,
                self.log_path,
                {"canary_events": self.canary_events, "guardrail_values": values},
            )
            self.decisions.append(record)
            logger.info("challenger_promoted", champion=record["champion"])
            return record
        return None


# ============= Service =============


def prometheus_query(base_url: str) -> Callable[[str], float | None]:
    """Return `query(promql)` evaluating an instant query; None when there is no sample."""
    import httpx

    def query(promql: str) -> float | None:
        try:
            response = httpx.get(f"{base_url}/api/v1/query", params={"query": promql}, timeout=10)
            response.raise_for_status()
            result = response.json()["data"]["result"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning("prometheus_query_failed", error=str(exc))
            return None
        if not result:
            return None
        value = float(result[0]["value"][1])
        return value if np.isfinite(value) else None

    return query


class RolloutService:
    """Kafka wiring for the rollout controller (runs in the orchestrator process)."""

    def __init__(self, dataset: str):
        from confluent_kafka import Consumer
        from redis import Redis

        from driftguard.common.kafka import consumer_config
        from driftguard.common.mlflow_client import MlflowRegistry
        from driftguard.data.schemas import topic_name
        from driftguard.data.serde import TopicSerde, registry_client

        settings = get_settings()
        self.dataset = dataset
        self.controller = RolloutController(
            dataset=dataset,
            config=load_config(settings.config_path("rollout.yaml")).data,
            registry=MlflowRegistry(settings.mlflow_tracking_uri),
            redis=Redis.from_url(settings.redis_url, decode_responses=True),
            log_path=settings.promotions_log_path,
            query=prometheus_query(settings.prometheus_url),
        )
        self.consumer = Consumer(
            consumer_config(
                settings.kafka_bootstrap_servers, f"driftguard-rollout-{dataset}", "rollout"
            )
        )
        client = registry_client(settings.schema_registry_url)
        self.topics = {
            topic_name("predictions"): TopicSerde(client, "predictions"),
            topic_name("labels", dataset): TopicSerde(client, "labels", dataset),
        }
        self.consumer.subscribe(list(self.topics))
        self.decode_failures = 0
        self._running = True

    def healthy(self) -> bool:
        """Liveness."""
        return self._running

    def stop(self, *_args: object) -> None:
        """Stop after the current poll."""
        self._running = False

    def run(self) -> None:
        """Consume predictions and labels and step the controller."""
        from driftguard.data.schemas import topic_name

        preds_t = topic_name("predictions")
        while self._running:
            predictions, labels = [], []
            for message in self.consumer.consume(num_messages=2000, timeout=1.0):
                if message.error():
                    continue
                try:
                    record = self.topics[message.topic()].deserialize(message.value())
                except Exception as exc:  # noqa: BLE001
                    # one poison record or a Schema Registry blip must not end the controller
                    self.decode_failures += 1
                    logger.warning("rollout_decode_skip", error=type(exc).__name__)
                    continue
                if message.topic() == preds_t:
                    if record["dataset"] == self.dataset:
                        predictions.append(record)
                else:
                    labels.append(record)
            self.controller.on_predictions(predictions)
            self.controller.on_labels(labels)
            try:
                self.controller.step()
            except Exception as exc:  # noqa: BLE001
                # registry or Prometheus errors retry on the next loop; the champion keeps serving
                logger.error("rollout_step_failed", error=str(exc))
            if predictions or labels:
                self.consumer.commit(asynchronous=True)
        self.consumer.close()


def main() -> None:
    """Run the rollout controller alone (normally it runs inside the orchestrator)."""
    service = RolloutService(get_settings().dataset)
    signal.signal(signal.SIGTERM, service.stop)
    service.run()


if __name__ == "__main__":
    main()
