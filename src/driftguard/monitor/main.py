"""Monitor service: windowed drift signals, policy decisions, drift events (SPEC 5)."""

import signal
import time
from collections import OrderedDict
from typing import Any

import pandas as pd

from driftguard.common.config import load_config
from driftguard.common.logging import configure_logging, get_logger
from driftguard.common.metrics import (
    ADWIN_DETECTIONS,
    DATA_QUALITY_FAILURES,
    DRIFT_EVENTS,
    DRIFT_SHARE,
    KSWIN_DETECTIONS,
    LABELED_SINCE_PROMOTION,
    LATE_EVENTS,
    PSI_MAX,
    PSI_SHARE,
    RETRAINING_PAUSED,
    SCORE_PSI,
    WINDOWS_CLOSED,
    WINDOWS_FLAGGED,
    start_metrics_server,
)
from driftguard.common.settings import get_settings
from driftguard.data.registry import dataset_config, feature_names
from driftguard.monitor.detectors import AdwinErrorDetector, KswinScoreDetector
from driftguard.monitor.policy import DriftPolicy, PolicyDecision
from driftguard.monitor.psi import psi_signals
from driftguard.monitor.reference import Reference
from driftguard.monitor.windows import ClosedWindow, CountWindows, MonthWindows

# ============= Constants =============

logger = get_logger("monitor")
PAUSE_KEY = "dg:retrain:paused:{dataset}"
PAUSED_BY_DATA_QUALITY = "data_quality"
MAX_PENDING = 500_000


# ============= Core =============


class MonitorCore:
    """
    Transport-free monitor logic: windows, signals, policy, pause flag.

    Parameters
    ----------
    dataset : str
        Dataset name.
    policy_config : dict
        `configs/drift_policy.yaml` content.
    redis_client : redis.Redis or compatible
        Holds the retraining pause flag.
    gx_validator : object or None
        Object with `validate(frame) -> report` (`report.success`, `report.failed`).
    window_mode : str
        `count` (CICIDS2017) or `month` (Kyoto).
    """

    def __init__(
        self,
        dataset: str,
        policy_config: dict[str, Any],
        redis_client,
        gx_validator=None,
        window_mode: str = "count",
    ):
        self.dataset = dataset
        self.config = policy_config
        self.policy = DriftPolicy(policy_config)
        self.redis = redis_client
        self.gx = gx_validator
        self.features = feature_names(dataset)
        self.windows = (
            CountWindows(int(policy_config["window_size"]))
            if window_mode == "count"
            else MonthWindows(int(policy_config["month_close_after_events"]))
        )
        adwin, kswin = policy_config["adwin"], policy_config["kswin"]
        self.adwin = AdwinErrorDetector(float(adwin["delta"]))
        self.kswin = KswinScoreDetector(
            float(kswin["alpha"]), int(kswin["window_size"]), int(kswin["stat_size"])
        )
        self.reference: Reference | None = None
        # predictions awaiting labels and labels awaiting predictions, bounded and oldest-first
        self._pending_predictions: OrderedDict[str, int] = OrderedDict()
        self._pending_labels: OrderedDict[str, int] = OrderedDict()
        self.decisions: list[PolicyDecision] = []
        self._late_seen = 0

    # ----- inputs -----

    def set_reference(self, reference: Reference, o_promotion: bool) -> None:
        """Install a new reference; a promotion resets policy counters and ADWIN."""
        self.reference = reference
        if o_promotion:
            self.policy.note_promotion()
            self.adwin.reset()
            self.kswin.reset()

    def on_flows(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add decoded flow records; returns any drift events emitted."""
        self.windows.add_flows(
            [(int(r["event_time"]), [float(r[name]) for name in self.features]) for r in records]
        )
        return self._close_windows()

    def on_predictions(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add champion scores and join decisions with pending labels for ADWIN."""
        self.windows.add_scores([(int(r["event_time"]), float(r["score"])) for r in records])
        errors = []
        for r in records:
            label = self._pending_labels.pop(r["event_id"], None)
            if label is None:
                self._remember(self._pending_predictions, r["event_id"], int(r["decision"]))
            else:
                errors.append(int(int(r["decision"]) != label))
        self._feed_adwin(errors)
        return self._close_windows()

    def on_labels(self, records: list[dict[str, Any]]) -> None:
        """Count labels and feed the champion's 0/1 error to ADWIN."""
        self.policy.add_labels(len(records))
        LABELED_SINCE_PROMOTION.labels(dataset=self.dataset).set(
            self.policy.labeled_since_promotion
        )
        errors = []
        for r in records:
            decision = self._pending_predictions.pop(r["event_id"], None)
            if decision is None:
                self._remember(self._pending_labels, r["event_id"], int(r["label"]))
            else:
                errors.append(int(decision != int(r["label"])))
        self._feed_adwin(errors)

    @staticmethod
    def _remember(store: OrderedDict, key: str, value: int) -> None:
        store[key] = value
        if len(store) > MAX_PENDING:
            store.popitem(last=False)

    def _feed_adwin(self, errors: list[int]) -> None:
        if errors:
            detections = self.adwin.update(errors)
            if detections:
                ADWIN_DETECTIONS.labels(dataset=self.dataset).inc(detections)

    # ----- windows -----

    def _close_windows(self) -> list[dict[str, Any]]:
        if isinstance(self.windows, MonthWindows) and self.windows.late_events > self._late_seen:
            LATE_EVENTS.labels(dataset=self.dataset).inc(self.windows.late_events - self._late_seen)
            self._late_seen = self.windows.late_events
        return [event for window in self.windows.ready() if (event := self._evaluate(window))]

    def compute_signals(self, window: ClosedWindow) -> dict[str, float]:
        """Compute every SPEC 5.2 signal for one window."""
        ref = self.reference
        index = [self.features.index(name) for name in ref.features]
        current = window.features[:, index]
        signals = psi_signals(
            ref.psi_profile,
            current,
            float(self.config["psi_share_feature_threshold"]),
            float(self.config["psi_floor"]),
        )
        signals["drift_share"] = ref.evidently.share(pd.DataFrame(current, columns=ref.features))
        signals["score_psi"] = ref.score_psi.psi(window.scores)
        kswin_signal, kswin_detections = self.kswin.window_signal(window.scores)
        if kswin_detections:
            KSWIN_DETECTIONS.labels(dataset=self.dataset).inc(kswin_detections)
        signals["kswin_score"] = kswin_signal
        signals["adwin_error"] = self.adwin.consume()
        return signals

    def _evaluate(self, window: ClosedWindow) -> dict[str, Any] | None:
        WINDOWS_CLOSED.labels(dataset=self.dataset).inc()
        if self.reference is None:
            logger.warning("window_skipped_no_reference", window_id=window.window_id)
            return None
        signals = self.compute_signals(window)
        gx_ok = True
        if self.gx is not None:
            report = self.gx.validate(pd.DataFrame(window.features, columns=self.features))
            gx_ok = report.success
            if not gx_ok:
                signals["gx_failure"] = 1.0
                logger.warning(
                    "data_quality_failed", window_id=window.window_id, failed=report.failed[:10]
                )
        self._publish_gauges(signals)
        decision = self.policy.evaluate(window.window_id, signals, gx_ok)
        self.decisions.append(decision)
        self._apply_pause(gx_ok)
        if decision.flagged:
            WINDOWS_FLAGGED.labels(dataset=self.dataset).inc()
        logger.info(
            "window_closed",
            window_id=window.window_id,
            flagged=decision.flagged,
            reasons=decision.reasons,
            blocked_by=decision.blocked_by,
            emit=decision.emit,
            **{k: round(v, 4) for k, v in signals.items()},
        )
        if not decision.emit:
            return None
        DRIFT_EVENTS.labels(dataset=self.dataset, severity=decision.severity).inc()
        return {
            "dataset": self.dataset,
            "window_id": window.window_id,
            "detected_at": int(time.time() * 1000),
            "signals": {k: float(v) for k, v in signals.items()},
            "severity": decision.severity,
            "policy_version": self.policy.version,
            "window_end_ms": window.end_event_time_ms,
        }

    def _publish_gauges(self, signals: dict[str, float]) -> None:
        PSI_MAX.labels(dataset=self.dataset).set(signals["psi_max"])
        PSI_SHARE.labels(dataset=self.dataset).set(signals["psi_share"])
        DRIFT_SHARE.labels(dataset=self.dataset).set(signals["drift_share"])
        SCORE_PSI.labels(dataset=self.dataset).set(signals["score_psi"])

    def _apply_pause(self, gx_ok: bool) -> None:
        """Pause retraining on a GX failure; lift a data-quality pause on the next clean window."""
        key = PAUSE_KEY.format(dataset=self.dataset)
        if not gx_ok:
            DATA_QUALITY_FAILURES.labels(dataset=self.dataset).inc()
            self.redis.set(key, PAUSED_BY_DATA_QUALITY)
            RETRAINING_PAUSED.labels(dataset=self.dataset).set(1)
            return
        if self.redis.get(key) == PAUSED_BY_DATA_QUALITY:
            # an operator pause (any other value) is never lifted automatically
            self.redis.delete(key)
        RETRAINING_PAUSED.labels(dataset=self.dataset).set(1 if self.redis.get(key) else 0)


# ============= Service =============


def drift_record(event: dict[str, Any]) -> dict[str, Any]:
    """Strip fields that are not part of the `drift-events-value` schema."""
    return {k: v for k, v in event.items() if k != "window_end_ms"} | {
        "signals": {**event["signals"], "window_end_ms": float(event["window_end_ms"])}
    }


class MonitorService:
    """Kafka wiring: consumes flows, predictions, and labels; produces drift events."""

    def __init__(self, dataset: str):
        from confluent_kafka import Consumer, Producer
        from redis import Redis

        from driftguard.common.kafka import consumer_config, producer_config
        from driftguard.common.mlflow_client import MlflowRegistry
        from driftguard.data.schemas import topic_name
        from driftguard.data.serde import TopicSerde, registry_client
        from driftguard.monitor.reference import ReferenceProvider
        from driftguard.training.validate import GXValidator, load_bounds

        self.settings = get_settings()
        self.dataset = dataset
        policy = load_config(self.settings.config_path("drift_policy.yaml")).data
        data_cfg = dataset_config(dataset)
        boot = self.settings.kafka_bootstrap_servers
        self.consumer = Consumer(
            consumer_config(boot, f"driftguard-monitor-{dataset}", f"monitor-{dataset}")
        )
        self.producer = Producer(producer_config(boot, f"monitor-{dataset}"))
        client = registry_client(self.settings.schema_registry_url)
        self.serdes = {
            topic_name("flows", dataset): TopicSerde(client, "flows", dataset),
            topic_name("predictions"): TopicSerde(client, "predictions"),
            topic_name("labels", dataset): TopicSerde(client, "labels", dataset),
        }
        self.drift_serde = TopicSerde(client, "drift")
        self.redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
        gx_cfg = data_cfg.get("gx", {})
        bounds = load_bounds(self.settings.config_path("data", f"{dataset}_gx_bounds.json"))
        gx = GXValidator(
            bounds,
            float(gx_cfg.get("mostly", 0.99)),
            int(gx_cfg.get("min_window_rows", 100)),
            name=f"{dataset}_window",
        )
        mode = data_cfg.get("monitor_windows", "count")
        self.core = MonitorCore(dataset, policy, self.redis, gx, mode)
        self.references = ReferenceProvider(
            MlflowRegistry(self.settings.mlflow_tracking_uri),
            dataset,
            feature_names(dataset),
            int(policy["psi_bins"]),
            int(policy["evidently_max_rows"]),
        )
        self._running = True

    def healthy(self) -> bool:
        """Liveness for `/healthz`."""
        return self._running

    def stop(self, *_args: object) -> None:
        """Stop after the current poll."""
        self._running = False

    def _refresh_reference(self) -> None:
        changed = self.references.refresh()
        if self.references.current is not None and (changed or self.core.reference is None):
            self.core.set_reference(self.references.current, o_promotion=changed)
            logger.info(
                "reference_loaded", version=self.references.current.version, promotion=changed
            )

    def run(self) -> None:
        """Poll the three topics and emit drift events."""
        from driftguard.data.schemas import topic_name

        self.consumer.subscribe(list(self.serdes))
        flows_t, preds_t = topic_name("flows", self.dataset), topic_name("predictions")
        last_ref = 0.0
        while self._running:
            if time.monotonic() - last_ref > 30:
                self._refresh_reference()
                last_ref = time.monotonic()
            grouped: dict[str, list[dict]] = {topic: [] for topic in self.serdes}
            for message in self.consumer.consume(num_messages=2000, timeout=1.0):
                if message.error():
                    continue
                try:
                    record = self.serdes[message.topic()].deserialize(message.value())
                except Exception as exc:  # noqa: BLE001
                    # malformed flows are already dead-lettered by the consumer; skip here
                    logger.debug("monitor_decode_skip", error=type(exc).__name__)
                    continue
                if message.topic() == preds_t and record["dataset"] != self.dataset:
                    continue
                grouped[message.topic()].append(record)
            events = self.core.on_flows(grouped[flows_t])
            events += self.core.on_predictions(grouped[preds_t])
            self.core.on_labels(grouped[topic_name("labels", self.dataset)])
            for event in events:
                self.producer.produce(
                    self.drift_serde.topic,
                    key=self.dataset.encode(),
                    value=self.drift_serde.serialize(drift_record(event)),
                )
                logger.info(
                    "drift_event_emitted", window_id=event["window_id"], severity=event["severity"]
                )
            self.producer.poll(0)
            if any(grouped.values()):
                self.consumer.commit(asynchronous=True)
        self.producer.flush(10)
        self.consumer.close()


def main() -> None:
    """Entry point: `python -m driftguard.monitor.main`."""
    settings = get_settings()
    configure_logging("monitor", settings.log_level)
    service = MonitorService(settings.dataset)
    signal.signal(signal.SIGTERM, service.stop)
    signal.signal(signal.SIGINT, service.stop)
    start_metrics_server(settings.metrics_port, service.healthy)
    service.run()


if __name__ == "__main__":
    main()
