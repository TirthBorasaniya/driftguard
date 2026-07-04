"""Kafka consumer: validate, compute features, infer, log, drift-detect, commit offset."""

import json
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from confluent_kafka import Consumer, KafkaError, Producer
from pydantic import ValidationError

from src.config import (
    DB_PATH,
    FEATURE_COLS,
    PRODUCTION_MODEL_PATH,
    REFERENCE_FILE,
    THRESHOLD_PATH,
    settings,
)
from src.consumer.dlq_handler import send_to_dlq
from src.consumer.schemas import NetworkFlowEvent
from src.features.engineering import compute_features
from src.monitoring.drift_detector import StreamingDriftDetector

# ============= Setup =============


def load_inference_artifacts():
    """Load model and decision threshold for inference."""
    import json as _json

    model = joblib.load(PRODUCTION_MODEL_PATH)

    threshold = 0.5
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            threshold = float(_json.load(f)["threshold"])

    return model, threshold


def build_feature_vector(feature_dict: dict) -> np.ndarray:
    """
    Convert a computed feature dict into a model feature vector.

    Parameters
    ----------
    feature_dict : dict
        Output of compute_features for a single network flow event.

    Returns
    -------
    features : np.ndarray
        Shape (1, n_features), ordered by FEATURE_COLS.
    """
    feature_values = [float(feature_dict.get(col, 0.0)) for col in FEATURE_COLS]
    return np.array([feature_values])


def log_prediction_sync(record: dict) -> None:
    """Write prediction to SQLite synchronously (called in consumer loop)."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """INSERT INTO predictions
               (event_id, anomaly_score, is_anomaly, threshold, model_version, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                record["event_id"],
                record["anomaly_score"],
                int(record["is_anomaly"]),
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
    2. Validate with NetworkFlowEvent Pydantic model
    3. On ValidationError: route to DLQ, continue
    4. Compute features via feature engineering (single source of truth)
    5. Run model inference, apply calibrated threshold
    6. Log prediction to SQLite (keyed by src_ip entity / event_id)
    7. Feed event into drift detector window buffer
    8. Manually commit offset (enable.auto.commit=False)
    """
    if not PRODUCTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {PRODUCTION_MODEL_PATH}. Run training first."
        )

    model, threshold = load_inference_artifacts()

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
                event = NetworkFlowEvent(**payload)
            except (json.JSONDecodeError, ValidationError, UnicodeDecodeError) as e:
                send_to_dlq(raw, str(e), dlq_producer)
                consumer.commit(asynchronous=False)
                continue

            event_dict = event.model_dump()

            # steps 4-5: compute features and infer
            feature_dict = compute_features(event_dict)
            features = build_feature_vector(feature_dict)
            anomaly_proba = float(model.predict_proba(features)[0, 1])
            is_anomaly = anomaly_proba >= threshold

            # step 6: log prediction keyed by the event entity (src_ip)
            log_prediction_sync({
                "event_id": event_dict["event_id"],
                "anomaly_score": anomaly_proba,
                "is_anomaly": is_anomaly,
                "threshold": threshold,
                "model_version": model_version,
                "timestamp": datetime.utcnow().isoformat(),
            })

            # step 7: feed drift detector with raw fields plus engineered features
            drift_detector.add_event({**event_dict, **feature_dict})

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
