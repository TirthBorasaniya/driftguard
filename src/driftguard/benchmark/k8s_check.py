"""In-cluster duplicate and loss check for E3b: produced event ids versus predictions.v1."""

import argparse
import json
import time
import uuid

from driftguard.common.kafka import consumer_config
from driftguard.common.settings import get_settings
from driftguard.data.loader import load_stream_frame
from driftguard.data.schemas import topic_name
from driftguard.data.serde import TopicSerde, registry_client

# ============= Check =============


def expected_ids(
    dataset: str,
    slices: list[tuple[int, int]],
    source: str = "dataset",
    n_events: int | None = None,
) -> set[str]:
    """Event ids the producer jobs emitted, from the same dataset slices they replayed."""
    producer_config = {"synthetic": {"n_events": n_events}, "seed": 42} if n_events else None
    frame = load_stream_frame(dataset, source, producer_config)
    ids: set[str] = set()
    for start, count in slices:
        ids.update(frame["event_id"].iloc[start : start + count].astype(str))
    return ids


def count_predictions(ids: set[str], idle_s: float, timeout_s: float) -> tuple[int, int]:
    """Read predictions.v1 from the beginning; return (records for ids, distinct ids seen)."""
    from confluent_kafka import Consumer

    settings = get_settings()
    serde = TopicSerde(registry_client(settings.schema_registry_url), "predictions")
    consumer = Consumer(
        consumer_config(
            settings.kafka_bootstrap_servers, f"e3b-check-{uuid.uuid4().hex}", "e3b-check"
        )
    )
    consumer.subscribe([topic_name("predictions")])
    total, seen = 0, set()
    last, deadline = time.monotonic(), time.monotonic() + timeout_s
    while time.monotonic() < deadline and time.monotonic() - last < idle_s:
        for message in consumer.consume(num_messages=5000, timeout=1.0):
            if message.error():
                continue
            last = time.monotonic()
            event_id = serde.deserialize(message.value())["event_id"]
            if event_id in ids:
                total += 1
                seen.add(event_id)
    consumer.close()
    return total, len(seen)


def main() -> None:
    """Print one JSON line with produced, predicted, duplicate, and lost counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cicids2017")
    parser.add_argument("--slices", required=True, help="start:count,start:count,...")
    parser.add_argument("--source", choices=["dataset", "synthetic"], default="dataset")
    parser.add_argument("--n-events", type=int, help="synthetic stream size")
    parser.add_argument("--idle-s", type=float, default=60)
    parser.add_argument("--timeout-s", type=float, default=1800)
    args = parser.parse_args()
    slices = [tuple(int(x) for x in part.split(":")) for part in args.slices.split(",")]
    ids = expected_ids(args.dataset, slices, args.source, args.n_events)
    total, distinct = count_predictions(ids, args.idle_s, args.timeout_s)
    print(
        json.dumps(
            {
                "e3b_check": {
                    "produced_distinct": len(ids),
                    "prediction_records": total,
                    "predicted_distinct": distinct,
                    "duplicates": total - distinct,
                    "lost": len(ids) - distinct,
                }
            }
        )
    )


if __name__ == "__main__":
    main()
