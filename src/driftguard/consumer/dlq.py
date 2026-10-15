"""Dead-letter queue routing with error headers (SPEC 3.1, 4.1)."""

from collections.abc import Callable

from driftguard.common.metrics import EVENTS_DLQ

# ============= Constants =============

ERROR_TYPES = ("decode", "validation", "scoring_unavailable")
MAX_ERROR_MSG = 1000


# ============= Routing =============


def dlq_headers(
    error_type: str, error_msg: str, source_topic: str, source_offset: int
) -> list[tuple[str, bytes]]:
    """
    Build DLQ headers.

    Parameters
    ----------
    error_type : str
        One of `decode`, `validation`, `scoring_unavailable`.
    error_msg : str
        Error description, truncated; never contains feature payloads or IPs.
    source_topic : str
        Topic the message came from.
    source_offset : int
        Offset of the original message.

    Returns
    -------
    headers : list of (str, bytes)
    """
    if error_type not in ERROR_TYPES:
        raise ValueError(f"unknown error_type {error_type!r}")
    return [
        ("error_type", error_type.encode()),
        ("error_msg", error_msg[:MAX_ERROR_MSG].encode()),
        ("source_topic", source_topic.encode()),
        ("source_offset", str(source_offset).encode()),
    ]


class DeadLetterQueue:
    """
    Sends original message bytes to `flows.dlq` with error headers.

    Parameters
    ----------
    produce : callable
        `produce(topic, key=..., value=..., headers=...)`, a confluent_kafka Producer method.
    dataset : str
        Dataset label for metrics.
    topic : str
        DLQ topic.
    """

    def __init__(self, produce: Callable[..., None], dataset: str, topic: str = "flows.dlq"):
        self._produce = produce
        self.dataset = dataset
        self.topic = topic

    def send(
        self,
        key: bytes | None,
        value: bytes | None,
        error_type: str,
        error_msg: str,
        source_topic: str,
        source_offset: int,
    ) -> None:
        """Route one message to the DLQ and count it."""
        self._produce(
            self.topic,
            key=key,
            value=value,
            headers=dlq_headers(error_type, error_msg, source_topic, source_offset),
        )
        EVENTS_DLQ.labels(dataset=self.dataset, error_type=error_type).inc()
