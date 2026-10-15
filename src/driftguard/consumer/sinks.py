"""Feature sinks (Feast online, partitioned offline parquet) and the delayed-label sink."""

import threading
import time
from collections.abc import Callable

import pandas as pd

from driftguard.common.logging import get_logger
from driftguard.features.offline import OfflineStore

# ============= Constants =============

logger = get_logger("consumer.sinks")


# ============= Frames =============


def feature_frame(records: list[dict], features: list[str]) -> pd.DataFrame:
    """
    Build the per-event feature rows written online and offline.

    The frame has `event_id`, `event_timestamp` (the flow's event time, UTC, set from the
    event and never from ingestion time), and the allowlisted features as float32. IPs are not
    included.
    """
    frame = pd.DataFrame.from_records(
        [{name: record[name] for name in features} for record in records], columns=features
    ).astype("float32")
    frame.insert(0, "event_id", [record["event_id"] for record in records])
    frame.insert(
        1,
        "event_timestamp",
        pd.to_datetime([record["event_time"] for record in records], unit="ms", utc=True),
    )
    return frame


# ============= Feature sink =============


class FeatureSink:
    """
    Writes each batch's features to the Feast online store and appends them to the offline store.

    Parameters
    ----------
    dataset : str
        Dataset name.
    offline : OfflineStore or None
        Offline parquet store; None disables offline writes.
    online_write : callable or None
        `online_write(feature_view_name, frame)`, normally `FeatureStore.write_to_online_store`.
    """

    def __init__(
        self,
        dataset: str,
        offline: OfflineStore | None,
        online_write: Callable[[str, pd.DataFrame], None] | None,
    ):
        self.dataset = dataset
        self.offline = offline
        self.online_write = online_write
        self.view_name = f"flow_features_{dataset}"

    def write(self, frame: pd.DataFrame) -> None:
        """Write one batch synchronously; the batch is not committed until this returns."""
        if frame.empty:
            return
        if self.online_write is not None:
            self.online_write(self.view_name, frame)
        if self.offline is not None:
            self.offline.append(self.dataset, "flow_features", frame)


# ============= Label sink =============


class LabelSink:
    """
    Persists delayed labels to the offline store; runs as a thread in the consumer service.

    Offsets are committed only after the buffered rows are written, so labels are at-least-once;
    the training window de-duplicates labels by `event_id`.

    Parameters
    ----------
    consumer : confluent_kafka.Consumer
        Consumer subscribed to `labels.<dataset>.v1` in its own group.
    deserialize : callable
        Label payload decoder.
    offline : OfflineStore
        Destination store.
    dataset : str
        Dataset name.
    flush_rows : int
        Buffer size that triggers a write.
    flush_s : float
        Maximum seconds between writes.
    """

    def __init__(
        self,
        consumer,
        deserialize,
        offline: OfflineStore,
        dataset: str,
        flush_rows: int = 20000,
        flush_s: float = 10.0,
    ):
        self.consumer = consumer
        self.deserialize = deserialize
        self.offline = offline
        self.dataset = dataset
        self.flush_rows = flush_rows
        self.flush_s = flush_s
        self._buffer: list[dict] = []
        self._stop = threading.Event()
        self.written = 0

    def _flush(self) -> None:
        if not self._buffer:
            return
        frame = pd.DataFrame(self._buffer)
        frame["label_timestamp"] = pd.to_datetime(frame["label_time"], unit="ms", utc=True)
        self.offline.append(self.dataset, "labels", frame)
        self.consumer.commit(asynchronous=False)
        self.written += len(frame)
        self._buffer = []

    def run(self) -> None:
        """Consume until `stop()` is called."""
        last_flush = time.monotonic()
        while not self._stop.is_set():
            for message in self.consumer.consume(num_messages=1000, timeout=1.0):
                if message.error():
                    logger.warning("label_message_error", error=str(message.error()))
                    continue
                try:
                    self._buffer.append(self.deserialize(message.value()))
                except Exception as exc:  # noqa: BLE001
                    # a malformed label is logged and skipped; labels never block scoring
                    logger.error("label_decode_failed", offset=message.offset(), error=str(exc))
            due = time.monotonic() - last_flush >= self.flush_s
            if len(self._buffer) >= self.flush_rows or (due and self._buffer):
                self._flush()
                last_flush = time.monotonic()
        self._flush()

    def stop(self) -> None:
        """Signal the run loop to finish after a final flush."""
        self._stop.set()
