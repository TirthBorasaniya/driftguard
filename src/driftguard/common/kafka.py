"""Kafka client configuration and topic administration helpers."""

from confluent_kafka import KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

from driftguard.common.logging import get_logger
from driftguard.data.schemas import PARTITIONS, topic_name

# ============= Constants =============

logger = get_logger("kafka")


# ============= Client configs =============


def producer_config(bootstrap: str, client_id: str, **overrides: object) -> dict:
    """
    Producer settings: idempotent, acks=all, batched with lz4 compression.

    Parameters
    ----------
    bootstrap : str
        Bootstrap servers.
    client_id : str
        Client id for broker logs.
    overrides : dict
        Extra librdkafka settings.

    Returns
    -------
    config : dict
        librdkafka producer configuration.
    """
    config = {
        "bootstrap.servers": bootstrap,
        "client.id": client_id,
        "enable.idempotence": True,
        "acks": "all",
        "linger.ms": 20,
        "batch.num.messages": 10000,
        "compression.type": "lz4",
        "queue.buffering.max.messages": 500000,
    }
    config.update(overrides)
    return config


def consumer_config(bootstrap: str, group_id: str, client_id: str, **overrides: object) -> dict:
    """
    Consumer settings: manual commits, cooperative-sticky assignment (SPEC 4.1, 8.1).

    Parameters
    ----------
    bootstrap : str
        Bootstrap servers.
    group_id : str
        Consumer group.
    client_id : str
        Client id.
    overrides : dict
        Extra librdkafka settings.

    Returns
    -------
    config : dict
        librdkafka consumer configuration.
    """
    config = {
        "bootstrap.servers": bootstrap,
        "group.id": group_id,
        "client.id": client_id,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
        "partition.assignment.strategy": "cooperative-sticky",
        "session.timeout.ms": 45000,
        "max.poll.interval.ms": 300000,
    }
    config.update(overrides)
    return config


# ============= Administration =============


def desired_topics(datasets: tuple[str, ...]) -> dict[str, int]:
    """Return every topic name mapped to its partition count (SPEC 3.1)."""
    topics = {}
    for dataset in datasets:
        topics[topic_name("flows", dataset)] = PARTITIONS["flows"]
        topics[topic_name("labels", dataset)] = PARTITIONS["labels"]
    topics[topic_name("predictions")] = PARTITIONS["predictions"]
    topics[topic_name("drift")] = PARTITIONS["drift"]
    topics[topic_name("dlq")] = PARTITIONS["dlq"]
    return topics


def ensure_topics(bootstrap: str, datasets: tuple[str, ...], replication: int = 1) -> list[str]:
    """
    Create missing topics with the SPEC partition counts.

    Parameters
    ----------
    bootstrap : str
        Bootstrap servers.
    datasets : tuple of str
        Datasets whose flow and label topics are needed.
    replication : int
        Replication factor (1 for the single-broker stacks).

    Returns
    -------
    created : list of str
        Topics created by this call.
    """
    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = set(admin.list_topics(timeout=10).topics)
    wanted = desired_topics(datasets)
    new = [
        NewTopic(name, num_partitions=parts, replication_factor=replication)
        for name, parts in wanted.items()
        if name not in existing
    ]
    if not new:
        return []
    created = []
    for name, future in admin.create_topics(new).items():
        try:
            future.result()
            created.append(name)
        except KafkaException as exc:
            logger.error("topic_create_failed", topic=name, error=str(exc))
            raise
    logger.info("topics_created", topics=created)
    return created


# ============= Delivery tracking =============


class DeliveryError(RuntimeError):
    """One or more produced messages failed delivery."""


class DeliveryTracker:
    """
    Records per-message delivery failures reported by librdkafka.

    `Producer.flush` returns the number of messages still queued, not the number that failed, so a
    message that failed fast is invisible to it. Pass `tracker` as `on_delivery` to every
    `produce` call and call `raise_if_failed` after `flush`, before marking work done or
    committing offsets.
    """

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.delivered = 0

    def __call__(self, err, msg) -> None:
        if err is not None:
            self.failures.append(f"{msg.topic()}[{msg.partition()}]: {err}")
        else:
            self.delivered += 1

    def raise_if_failed(self) -> None:
        """Raise DeliveryError (and clear the record) when any delivery failed."""
        if self.failures:
            failures, self.failures = self.failures, []
            raise DeliveryError(f"{len(failures)} message(s) failed delivery: {failures[:3]}")
