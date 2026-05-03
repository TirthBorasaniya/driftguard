"""Kafka producer: chronological replay of stream split with optional drift injection."""

import argparse
import json
import time

import pandas as pd
from confluent_kafka import Producer

from src.config import DRIFT_SCENARIOS_DIR, STREAM_FILE, settings
from src.producer.drift_injector import inject_drift, load_scenario


def delivery_callback(err, msg) -> None:
    """Log delivery errors."""
    if err:
        print(f"Delivery failed: {err}")


def run_producer(o_drift: bool = False, delay_seconds: float = 0.05) -> None:
    """
    Stream transactions from data/processed/stream.parquet to Kafka.

    Sends each row chronologically. In drift mode, applies heavy_drift.json
    transformations to simulate distributional shift.

    Parameters
    ----------
    o_drift : bool
        If True, inject drift according to heavy_drift.json scenario.
    delay_seconds : float
        Pause between messages for demo pacing.
    """
    if not STREAM_FILE.exists():
        raise FileNotFoundError(
            f"Stream file not found: {STREAM_FILE}. Run preprocessing first."
        )

    producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})

    scenario = None
    if o_drift:
        scenario_path = DRIFT_SCENARIOS_DIR / "heavy_drift.json"
        scenario = load_scenario(scenario_path)
        print(f"Drift injection enabled: {scenario['name']}")
        print(f"  amt_multiplier={scenario.get('amt_multiplier')}, "
              f"state={scenario.get('state_concentration')}")

    df = pd.read_parquet(STREAM_FILE)
    print(f"Streaming {len(df):,} transactions to topic '{settings.kafka_topic}'")
    print("Press Ctrl+C to stop.\n")

    sent = 0
    try:
        for _, row in df.iterrows():
            record = row.to_dict()

            # convert non-serializable types
            for k, v in record.items():
                if hasattr(v, "item"):
                    record[k] = v.item()
                elif pd.isna(v) if not isinstance(v, str) else False:
                    record[k] = None

            if o_drift and scenario:
                record = inject_drift(record, scenario)

            producer.produce(
                topic=settings.kafka_topic,
                value=json.dumps(record).encode("utf-8"),
                callback=delivery_callback,
            )
            producer.poll(0)
            sent += 1

            if sent % 500 == 0:
                print(f"  Sent {sent:,} messages")

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        producer.flush()
        print(f"Producer done. Total sent: {sent:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud transaction Kafka producer")
    parser.add_argument("--drift", action="store_true", help="Enable drift injection")
    parser.add_argument("--delay", type=float, default=0.05, help="Seconds between messages")
    args = parser.parse_args()
    run_producer(o_drift=args.drift, delay_seconds=args.delay)
