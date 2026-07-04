"""Dead letter queue handler: route malformed Kafka messages to the network_flows.dlq topic."""

from confluent_kafka import Producer

from src.config import settings

DLQ_TOPIC = settings.kafka_dlq_topic
DLQ_ERROR_HEADER_KEY = "validation_error"


def route_to_dlq(
    producer: Producer,
    raw_message: bytes,
    validation_error: str,
    dlq_topic: str = DLQ_TOPIC,
) -> None:
    """
    Publish a malformed message to the dead letter queue with the validation
    error attached as a header.

    The original message bytes are forwarded unchanged. Never silently drops
    malformed events.

    Parameters
    ----------
    producer : Producer
        Initialized Confluent Kafka producer.
    raw_message : bytes
        The original malformed message payload.
    validation_error : str
        Description of why validation failed.
    dlq_topic : str
        Destination DLQ topic name.
    """
    producer.produce(
        topic=dlq_topic,
        value=raw_message,
        headers={DLQ_ERROR_HEADER_KEY: validation_error.encode("utf-8")},
    )
    producer.poll(0)
