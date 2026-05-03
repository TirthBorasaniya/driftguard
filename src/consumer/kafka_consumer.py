"""Kafka consumer: validate, engineer features, infer, log, drift-detect, commit offset."""

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from confluent_kafka import Consumer, KafkaError, Producer
from pydantic import ValidationError

from src.config import (
    CATEGORICAL_COLS,
    DB_PATH,
    ENCODERS_DIR,
    FEATURE_COLS,
    PRODUCTION_MODEL_PATH,
    REFERENCE_FILE,
    THRESHOLD_PATH,
    settings,
)
from src.consumer.dlq_handler import send_to_dlq
from src.consumer.schemas import TransactionEvent
from src.data.encoders import load_encoders
from src.features.engineering import engineer_single_event
from src.monitoring.drift_detector import StreamingDriftDetector


# ============= Setup =============


def load_inference_artifacts():
    """Load model, threshold, encoders for inference."""
    import json as _json

    model = joblib.load(PRODUCTION_MODEL_PATH)

    threshold = 0.5
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            threshold = float(_json.load(f)["threshold"])

    encoders = load_encoders(CATEGORICAL_COLS, ENCODERS_DIR)
    return model, threshold, encoders


def build_feature_vector(event: dict, encoders: dict) -> np.ndarray:
    """Convert a validated event dict into a model feature vector."""
    event = engineer_single_event(event)

    for col in CATEGORICAL_COLS:
        enc = encoders.get(col)
        if enc is not None:
            encoded = enc.transform(pd.Series([str(event.get(col, ""))]))
            event[col] = int(encoded.iloc[0])
        else:
            event[col] = -1

    feature_values = [
        float(event.get(col, -999) or -999)
        for col in FEATURE_COLS
    ]
    return np.array([feature_values])


def log_prediction_sync(record: dict) -> None:
    """Write prediction to SQLite synchronously (called in consumer loop)."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """INSERT INTO predictions
               (transaction_id, fraud_probability, is_fraud, threshold, model_version, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                record["transaction_id"],
                record["fraud_probability"],
                int(record["is_fraud"]),
                record["threshold"],
                record["model_version"],
                record["timestamp"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ============= Consumer Loop =============


def run_consumer() -> None:
    """
    Main Kafka consumer loop.

    Processing steps per message (in order):
    1. Deserialize JSON
    2. Validate with TransactionEvent Pydantic model
    3. On ValidationError: route to DLQ, continue
    4. Engineer features via engineering.py
    5. Run model inference, apply F2 threshold
    6. Log prediction to SQLite
    7. Feed event into drift detector window buffer
    8. Manually commit offset (enable.auto.commit=False)
    """
    if not PRODUCTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {PRODUCTION_MODEL_PATH}. Run training first."
        )

    model, threshold, encoders = load_inference_artifacts()

    reference_df = None
    if REFERENCE_FILE.exists():
        reference_df = pd.read_parquet(REFERENCE_FILE)

    drift_detector = StreamingDriftDetector(
        reference_df=reference_df,
        min_window=settings.drift_min_window,
    )

    consumer = Consumer({
        "bootstrap.servers": settings.kafka_bootstrap_servers,
        "group.id": settings.kafka_group_id,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([settings.kafka_topic])

    dlq_producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})

    model_version = datetime.fromtimestamp(
        PRODUCTION_MODEL_PATH.stat().st_mtime
    ).strftime("%Y%m%d-%H%M%S")

    print(f"Consumer started. Topic: {settings.kafka_topic}")
    processed = 0

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                print(f"Consumer error: {msg.error()}")
                continue

            raw = msg.value()

            # step 1-3: deserialize and validate
            try:
                payload = json.loads(raw.decode("utf-8"))
                event = TransactionEvent(**payload)
            except (json.JSONDecodeError, ValidationError, UnicodeDecodeError) as e:
                send_to_dlq(raw, str(e), dlq_producer)
                consumer.commit(asynchronous=False)
                continue

            event_dict = event.model_dump()

            # steps 4-5: engineer features and infer
            features = build_feature_vector(event_dict, encoders)
            fraud_proba = float(model.predict_proba(features)[0, 1])
            is_fraud = fraud_proba >= threshold

            # step 6: log prediction
            log_prediction_sync({
                "transaction_id": str(uuid.uuid4())[:12],
                "fraud_probability": fraud_proba,
                "is_fraud": is_fraud,
                "threshold": threshold,
                "model_version": model_version,
                "timestamp": datetime.utcnow().isoformat(),
            })

            # step 7: feed drift detector
            drift_detector.add_event(event_dict)

            # step 8: commit offset only after full processing
            consumer.commit(asynchronous=False)

            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed:,} messages")

    except KeyboardInterrupt:
        print("\nConsumer stopped.")
    finally:
        consumer.close()
        dlq_producer.flush()


if __name__ == "__main__":
    run_consumer()
