"""Dead letter queue handler: route malformed Kafka messages to transactions.dlq."""

from confluent_kafka import Producer

from src.config import settings


def send_to_dlq(
    raw_message_value: bytes,
    error: str,
    producer: Producer,
) -> None:
    """
    Publish a malformed message to the dead letter queue topic.

    The original message bytes are forwarded unchanged. The validation error
    is attached as a Kafka header so downstream consumers can inspect it.
    Never silently drops malformed events.

    Parameters
    ----------
    raw_message_value : bytes
        Original undecoded message payload.
    error : str
        Validation or deserialization error description.
    producer : Producer
        Confluent Kafka producer instance.
    """
    producer.produce(
        topic=settings.kafka_dlq_topic,
        value=raw_message_value,
        headers={"error": error.encode("utf-8")},
    )
    producer.poll(0)
