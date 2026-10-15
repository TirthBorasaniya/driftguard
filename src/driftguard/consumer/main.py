"""Scoring consumer service: decode, validate, dedupe, write features, score, log (SPEC 4.1)."""

import contextlib
import os
import signal
import socket
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from driftguard.common.config import load_config
from driftguard.common.logging import configure_logging, get_logger
from driftguard.common.metrics import (
    BATCH_SECONDS,
    CONSUMER_LAG,
    DUPLICATES,
    EVENTS_CONSUMED,
    SCORING_LATENCY,
    SCORING_UNAVAILABLE,
    SHADOW_FAILURES,
    STREAM_FEATURE_SECONDS,
    start_metrics_server,
)
from driftguard.common.settings import get_settings
from driftguard.consumer.dlq import DeadLetterQueue
from driftguard.consumer.idempotency import IdempotencyStore
from driftguard.consumer.scoring import SHADOW_MODE, ScoringClient, ScoringUnavailableError
from driftguard.consumer.sinks import FeatureSink, feature_frame
from driftguard.consumer.validation import FlowValidator, ValidationError
from driftguard.data.registry import feature_names

# ============= Constants =============

logger = get_logger("consumer")
StreamFeatureFn = Callable[[list[dict[str, Any]]], list[dict[str, float]]]


# ============= Batch processing =============


@dataclass
class Message:
    """Transport-neutral view of one Kafka message."""

    key: bytes | None
    value: bytes | None
    topic: str
    partition: int
    offset: int


@dataclass
class BatchOutcome:
    """Counts for one processed batch."""

    consumed: int = 0
    dlq: dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    scored: int = 0
    predictions: list[dict[str, Any]] = field(default_factory=list)


class BatchProcessor:
    """
    Processes one poll batch end to end; Kafka consumption and commits stay outside.

    Parameters
    ----------
    dataset : str
        Dataset name.
    deserialize : callable
        Flow payload decoder.
    validator : FlowValidator
        Schema-level validator.
    idempotency : IdempotencyStore
        Claim and done keys.
    sink : FeatureSink
        Online and offline feature writer.
    scorer : ScoringClient
        Serving router client.
    dlq : DeadLetterQueue
        Dead-letter router.
    emit_prediction : callable
        `emit_prediction(record)` produces to `predictions.v1`.
    flush : callable
        Flushes the producer; must return only after delivery.
    stream_features : callable or None
        Computes behavioral features for the new events (SPEC 4.3).
    batch_size : int
        Events per scoring request (SPEC: 256).
    """

    def __init__(
        self,
        dataset: str,
        deserialize: Callable[[bytes], dict[str, Any]],
        validator: FlowValidator,
        idempotency: IdempotencyStore,
        sink: FeatureSink,
        scorer: ScoringClient,
        dlq: DeadLetterQueue,
        emit_prediction: Callable[[dict[str, Any]], None],
        flush: Callable[[], None],
        stream_features: StreamFeatureFn | None = None,
        batch_size: int = 256,
        wait_poll_s: float = 0.5,
        behavior_sink=None,
    ):
        self.dataset = dataset
        self.deserialize = deserialize
        self.validator = validator
        self.idempotency = idempotency
        self.sink = sink
        self.scorer = scorer
        self.dlq = dlq
        self.emit_prediction = emit_prediction
        self.flush = flush
        self.stream_features = stream_features
        self.behavior_sink = behavior_sink
        self.batch_size = batch_size
        self.wait_poll_s = wait_poll_s
        self.features = feature_names(dataset)
        self._shadow_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shadow")

    def _decode_and_validate(
        self, messages: list[Message], outcome: BatchOutcome
    ) -> list[tuple[Message, dict[str, Any]]]:
        valid = []
        for message in messages:
            try:
                record = self.deserialize(message.value)
            except Exception as exc:  # noqa: BLE001
                # decoder failures (bad magic byte, unknown schema id, truncated Avro) go to the DLQ
                self._to_dlq(message, "decode", type(exc).__name__, outcome)
                continue
            try:
                self.validator.validate(record)
            except ValidationError as exc:
                self._to_dlq(message, "validation", str(exc), outcome)
                continue
            valid.append((message, record))
        return valid

    def _to_dlq(self, message: Message, error_type: str, msg: str, outcome: BatchOutcome) -> None:
        self.dlq.send(message.key, message.value, error_type, msg, message.topic, message.offset)
        outcome.dlq[error_type] = outcome.dlq.get(error_type, 0) + 1

    def _dedupe(self, valid, outcome: BatchOutcome) -> list[tuple[Message, dict[str, Any]]]:
        seen, unique = set(), []
        for message, record in valid:
            if record["event_id"] in seen:
                outcome.duplicates += 1
                continue
            seen.add(record["event_id"])
            unique.append((message, record))
        claim = self.idempotency.claim_with_wait(
            [record["event_id"] for _, record in unique], poll_s=self.wait_poll_s
        )
        outcome.duplicates += len(claim.duplicates)
        if claim.held:
            # still held after twice the lease: process anyway rather than lose the events
            logger.warning("idempotency_wait_exhausted", count=len(claim.held))
        skip = set(claim.duplicates)
        return [(m, r) for m, r in unique if r["event_id"] not in skip]

    def _vectors(self, records: list[dict[str, Any]], extra: list[dict[str, float]] | None):
        vectors = []
        for i, record in enumerate(records):
            vector = [float(record[name]) for name in self.features]
            if extra is not None:
                vector += [float(v) for v in extra[i].values()]
            vectors.append((record["event_id"], vector))
        return vectors

    def process(self, messages: list[Message]) -> BatchOutcome:
        """
        Run steps 1 to 6 of SPEC 4.1 on one poll batch.

        Predictions are produced and flushed, then the claimed ids are marked done. The caller
        commits offsets only after this returns, which makes processing at-least-once and the
        idempotency keys make it effectively once.
        """
        started = time.perf_counter()
        outcome = BatchOutcome(consumed=len(messages))
        EVENTS_CONSUMED.labels(dataset=self.dataset).inc(len(messages))
        valid = self._decode_and_validate(messages, outcome)
        fresh = self._dedupe(valid, outcome)
        DUPLICATES.labels(dataset=self.dataset).inc(outcome.duplicates)
        if not fresh:
            self.flush()
            return outcome

        # events are processed in event-time order so stream-feature windows see them in order
        fresh.sort(key=lambda item: (item[1]["event_time"], item[1]["event_id"]))
        records = [record for _, record in fresh]
        extra = None
        if self.stream_features is not None:
            sf_start = time.perf_counter()
            extra = self.stream_features(records)
            STREAM_FEATURE_SECONDS.labels(dataset=self.dataset).observe(
                time.perf_counter() - sf_start
            )
        frame = feature_frame(records, self.features)
        self.sink.write(frame)
        if extra and self.behavior_sink is not None:
            behavior = frame[["event_id", "event_timestamp"]].copy()
            for name in extra[0]:
                behavior[name] = [float(row[name]) for row in extra]
            behavior[list(extra[0])] = behavior[list(extra[0])].astype("float32")
            self.behavior_sink.write(behavior)

        vectors = self._vectors(records, extra)
        names = self.features + (list(extra[0].keys()) if extra else [])
        by_id = {record["event_id"]: (message, record) for message, record in fresh}
        done: list[str] = []
        for start in range(0, len(vectors), self.batch_size):
            chunk = vectors[start : start + self.batch_size]
            self._score_chunk(chunk, names, by_id, outcome, done)
        self.flush()
        self.idempotency.mark_done(done)
        BATCH_SECONDS.labels(dataset=self.dataset).observe(time.perf_counter() - started)
        return outcome

    def _score_chunk(self, chunk, names, by_id, outcome: BatchOutcome, done: list[str]) -> None:
        shadow_future = None
        if self.scorer.last_mode == SHADOW_MODE:
            shadow_future = self._shadow_pool.submit(self.scorer.score_shadow, chunk, names)
        try:
            response = self.scorer.score(chunk, names)
        except ScoringUnavailableError as exc:
            SCORING_UNAVAILABLE.labels(dataset=self.dataset).inc()
            ids = [eid for eid, _ in chunk]
            for eid in ids:
                message, _ = by_id[eid]
                self._to_dlq(message, "scoring_unavailable", str(exc)[:200], outcome)
            self.idempotency.release(ids)
            return
        shadow = None
        if shadow_future is not None:
            try:
                shadow = shadow_future.result(timeout=self.scorer.shadow_timeout_s)
            except Exception:  # noqa: BLE001
                # shadow scoring never fails a batch; the champion answer is already final
                SHADOW_FAILURES.labels(dataset=self.dataset).inc()
        for result in response.results:
            SCORING_LATENCY.labels(dataset=self.dataset, alias=result["model_alias"]).observe(
                response.latency_ms / 1000.0
            )
            _, record = by_id[result["event_id"]]
            prediction = {
                "event_id": result["event_id"],
                "dataset": self.dataset,
                "event_time": int(record["event_time"]),
                "model_alias": result["model_alias"],
                "model_version": str(result["model_version"]),
                "score": float(result["score"]),
                "decision": int(result["decision"]),
                "threshold": float(result["threshold"]),
                "latency_ms": float(response.latency_ms),
                "shadow_score": shadow["scores"].get(result["event_id"]) if shadow else None,
                "shadow_version": shadow["model_version"] if shadow else None,
            }
            self.emit_prediction(prediction)
            outcome.predictions.append(prediction)
            done.append(result["event_id"])
        outcome.scored += len(response.results)


# ============= Service =============


class ConsumerService:
    """Wires Kafka, Redis, Feast, and the serving client around a BatchProcessor."""

    def __init__(self, dataset: str, config_path: str | None = None):
        from confluent_kafka import Consumer, Producer
        from redis import Redis

        from driftguard.common.kafka import consumer_config, producer_config
        from driftguard.data.schemas import topic_name
        from driftguard.data.serde import TopicSerde, registry_client
        from driftguard.features.offline import OfflineStore

        self.settings = get_settings()
        self.dataset = dataset
        self.config = load_config(config_path or self.settings.config_path("consumer.yaml"))
        cfg = self.config
        owner = self.settings.instance_id or f"{socket.gethostname()}-{os.getpid()}"
        self.owner = f"{owner}-{uuid.uuid4().hex[:6]}"
        boot = self.settings.kafka_bootstrap_servers

        self.consumer = Consumer(consumer_config(boot, cfg["group_id"], f"consumer-{self.owner}"))
        self.topic = topic_name("flows", dataset)
        self.producer = Producer(producer_config(boot, f"consumer-{self.owner}"))
        client = registry_client(self.settings.schema_registry_url)
        flow_serde = TopicSerde(client, "flows", dataset)
        self.pred_serde = TopicSerde(client, "predictions")
        self.redis = Redis.from_url(self.settings.redis_url, decode_responses=True)
        idem = cfg["idempotency"]
        offline = (
            OfflineStore(self.settings.offline_store_dir)
            if cfg["offline_sink"]["enabled"]
            else None
        )
        self.store = self._feast_store() if cfg["feast"]["online_write"] else None
        online_write = self.store.write_to_online_store if self.store is not None else None
        scoring = cfg["scoring"]
        self.processor = BatchProcessor(
            dataset=dataset,
            deserialize=flow_serde.deserialize,
            validator=FlowValidator(dataset),
            idempotency=IdempotencyStore(
                self.redis, self.owner, idem["key_prefix"], idem["ttl_s"], idem["lease_s"]
            ),
            sink=FeatureSink(dataset, offline, online_write),
            scorer=ScoringClient(
                self.settings.serving_url,
                dataset,
                timeout_s=scoring["timeout_s"],
                retries=scoring["retries"],
                backoff_base_s=scoring["backoff_base_s"],
                backoff_max_s=scoring["backoff_max_s"],
                shadow_timeout_s=scoring["shadow_timeout_s"],
            ),
            dlq=DeadLetterQueue(self.producer.produce, dataset, topic_name("dlq")),
            emit_prediction=self._emit,
            flush=self._flush,
            stream_features=self._stream_features(),
            batch_size=scoring["batch_size"],
            wait_poll_s=idem["wait_poll_s"],
            behavior_sink=self._behavior_sink(offline),
        )
        self.label_sink = None
        if cfg["label_sink"]["enabled"] and offline is not None:
            from driftguard.consumer.sinks import LabelSink

            label_consumer = Consumer(
                consumer_config(boot, cfg["label_sink"]["group_id"], f"labels-{self.owner}")
            )
            label_consumer.subscribe([topic_name("labels", dataset)])
            self.label_sink = LabelSink(
                label_consumer,
                TopicSerde(client, "labels", dataset).deserialize,
                offline,
                dataset,
                cfg["label_sink"]["flush_rows"],
                cfg["label_sink"]["flush_s"],
            )
        self._running = True
        self._assigned: list = []

    def _feast_store(self):
        from driftguard.features.store import build_store, redis_online_store

        return build_store(
            self.settings.feast_repo_path,
            self.settings.data_dir / "feast" / "registry.db",
            redis_online_store(self.settings.redis_url),
        )

    def _behavior_sink(self, offline):
        from driftguard.data.registry import stream_features_enabled

        if not stream_features_enabled(self.dataset):
            return None
        from driftguard.streamfeatures.push import BehaviorSink, feast_online_pusher

        pusher = feast_online_pusher(self.store) if self.store is not None else None
        return BehaviorSink(self.dataset, offline, pusher)

    def _stream_features(self) -> StreamFeatureFn | None:
        from driftguard.data.registry import stream_features_enabled

        if not stream_features_enabled(self.dataset):
            return None
        from driftguard.streamfeatures.windows import BEHAVIOR_FEATURES, build_stream_feature_fn

        if self.settings.stream_features_mode == "off":
            # load-test mode: identical request shape, no Redis window work (E3 features off)
            logger.warning("stream_features_disabled_for_load_test")
            return lambda records: [dict.fromkeys(BEHAVIOR_FEATURES, 0.0) for _ in records]

        return build_stream_feature_fn(self.dataset, self.redis, self.settings)

    def _emit(self, prediction: dict[str, Any]) -> None:
        payload = self.pred_serde.serialize(prediction)
        while True:
            try:
                self.producer.produce(
                    self.pred_serde.topic, key=prediction["event_id"].encode(), value=payload
                )
                break
            except BufferError:
                self.producer.poll(0.1)
        self.producer.poll(0)

    def _flush(self) -> None:
        remaining = self.producer.flush(30)
        if remaining:
            raise RuntimeError(f"{remaining} messages not delivered; batch will be redelivered")

    def _on_assign(self, _consumer, partitions) -> None:
        self._assigned = list({(p.topic, p.partition) for p in partitions} | set(self._assigned))
        logger.info("partitions_assigned", partitions=[p.partition for p in partitions])

    def _on_revoke(self, _consumer, partitions) -> None:
        revoked = {(p.topic, p.partition) for p in partitions}
        self._assigned = [tp for tp in self._assigned if tp not in revoked]
        for _, partition in revoked:
            # a revoked partition's lag is now reported by its new owner; drop the stale series
            with contextlib.suppress(KeyError):
                CONSUMER_LAG.remove(self.dataset, str(partition))
        logger.info("partitions_revoked", partitions=[p.partition for p in partitions])

    def _update_lag(self) -> None:
        from confluent_kafka import TopicPartition

        if not self._assigned:
            return
        parts = [TopicPartition(t, p) for t, p in self._assigned]
        for tp in self.consumer.position(parts):
            _, high = self.consumer.get_watermark_offsets(tp, cached=True)
            lag = max(0, high - tp.offset) if tp.offset >= 0 else high
            CONSUMER_LAG.labels(dataset=self.dataset, partition=str(tp.partition)).set(lag)

    def healthy(self) -> bool:
        """Liveness for `/healthz`."""
        return self._running

    def stop(self, *_args: object) -> None:
        """Stop after the current batch."""
        self._running = False

    def run(self) -> None:
        """Poll, process, commit until stopped."""
        cfg = self.config
        self.consumer.subscribe([self.topic], on_assign=self._on_assign, on_revoke=self._on_revoke)
        if self.label_sink is not None:
            threading.Thread(target=self.label_sink.run, name="label-sink", daemon=True).start()
        last_lag = 0.0
        while self._running:
            raw = self.consumer.consume(
                num_messages=cfg["max_poll_records"], timeout=cfg["poll_timeout_s"]
            )
            batch = []
            for message in raw:
                if message.error():
                    logger.warning("kafka_message_error", error=str(message.error()))
                    continue
                batch.append(
                    Message(
                        message.key(),
                        message.value(),
                        message.topic(),
                        message.partition(),
                        message.offset(),
                    )
                )
            if batch:
                outcome = self.processor.process(batch)
                self.consumer.commit(asynchronous=False)
                logger.debug("batch_done", consumed=outcome.consumed, scored=outcome.scored)
            if time.monotonic() - last_lag >= cfg["lag_refresh_s"]:
                self._update_lag()
                last_lag = time.monotonic()
        if self.label_sink is not None:
            self.label_sink.stop()
        self.consumer.close()


def main() -> None:
    """Entry point: `python -m driftguard.consumer.main`."""
    settings = get_settings()
    configure_logging("consumer", settings.log_level)
    service = ConsumerService(settings.dataset)
    signal.signal(signal.SIGTERM, service.stop)
    signal.signal(signal.SIGINT, service.stop)
    start_metrics_server(settings.metrics_port, service.healthy)
    service.run()


if __name__ == "__main__":
    main()
