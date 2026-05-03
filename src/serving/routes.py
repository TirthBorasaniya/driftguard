"""FastAPI route handlers: /predict, /predict/explain, /health, and prediction history."""

import json
import time
import uuid
from datetime import datetime

import aiosqlite
import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from src.config import CATEGORICAL_COLS, DB_PATH, FEATURE_COLS, NUMERIC_COLS
from src.features.engineering import engineer_single_event
from src.serving.schemas import (
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)

router = APIRouter()


# ============= Prometheus Custom Metrics =============

try:
    from prometheus_client import Counter, Gauge

    PREDICTION_COUNTER = Counter(
        "model_predictions_total",
        "Total predictions by class",
        ["predicted_class"],
    )
    MODEL_CHAMPION_AUC_PR = Gauge(
        "model_champion_auc_pr",
        "AUC-PR of the currently loaded champion model",
    )
    MODEL_DRIFT_SCORE = Gauge(
        "model_drift_score",
        "Most recent drift share from Evidently batch report",
    )
    KAFKA_DLQ_MESSAGES = Counter(
        "kafka_dlq_messages_total",
        "Cumulative count of messages sent to dead letter queue",
    )
    _prom_available = True
except ImportError:
    _prom_available = False


# ============= Feature Preparation =============


def build_feature_vector(request: TransactionRequest, bundle) -> np.ndarray:
    """
    Convert a TransactionRequest into a model-ready feature vector.

    Applies the same transforms as preprocess.py via engineering.py,
    then encodes categoricals with SafeLabelEncoders.

    Parameters
    ----------
    request : TransactionRequest
    bundle : ModelBundle

    Returns
    -------
    features : np.ndarray
        Shape (1, n_features).
    """
    event = request.model_dump()
    event = engineer_single_event(event)

    # encode categoricals with fallback to -1 for unseen values
    for col in CATEGORICAL_COLS:
        enc = bundle.encoders.get(col)
        if enc is not None:
            import pandas as pd
            encoded = enc.transform(pd.Series([str(event.get(col, ""))]))
            event[col] = int(encoded.iloc[0])
        else:
            event[col] = -1

    feature_values = []
    for col in FEATURE_COLS:
        val = event.get(col, -999)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = -999.0
        feature_values.append(float(val))

    return np.array([feature_values])


# ============= Database Logging =============


async def log_prediction_async(record: dict) -> None:
    """Log a prediction record to the SQLite database."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(
            """INSERT INTO predictions
               (transaction_id, fraud_probability, is_fraud, threshold, model_version,
                timestamp, features_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                record["transaction_id"],
                record["fraud_probability"],
                int(record["is_fraud"]),
                record["threshold"],
                record["model_version"],
                record["timestamp"],
                record.get("features_json", "{}"),
            ),
        )
        await db.commit()


# ============= Endpoints =============


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Return API health and model readiness."""
    bundle = request.app.state.bundle
    return HealthResponse(
        status="healthy",
        model_loaded=bundle.model is not None,
        model_version=bundle.version,
        uptime_seconds=time.time() - request.app.state.start_time,
        total_predictions=request.app.state.prediction_count,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    transaction: TransactionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    """
    Predict fraud probability for a transaction.

    Returns probability and binary decision at the F2-optimized threshold.
    """
    bundle = request.app.state.bundle
    if bundle.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = build_feature_vector(transaction, bundle)
    fraud_proba = float(bundle.model.predict_proba(features)[0, 1])
    is_fraud = fraud_proba >= bundle.threshold

    if _prom_available:
        PREDICTION_COUNTER.labels(predicted_class="fraud" if is_fraud else "legit").inc()

    transaction_id = str(uuid.uuid4())[:12]
    timestamp = datetime.utcnow().isoformat()

    background_tasks.add_task(
        log_prediction_async,
        {
            "transaction_id": transaction_id,
            "fraud_probability": fraud_proba,
            "is_fraud": is_fraud,
            "threshold": bundle.threshold,
            "model_version": bundle.version,
            "timestamp": timestamp,
            "features_json": json.dumps({"amt": transaction.amt, "cc_num": transaction.cc_num}),
        },
    )
    request.app.state.prediction_count += 1

    return PredictionResponse(
        transaction_id=transaction_id,
        fraud_probability=round(fraud_proba, 6),
        is_fraud=is_fraud,
        threshold=bundle.threshold,
        model_version=bundle.version,
        timestamp=timestamp,
    )


@router.post("/predict/explain", response_model=ExplainResponse)
async def predict_explain(
    transaction: TransactionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> ExplainResponse:
    """
    Predict fraud probability with top-5 SHAP feature contributions.

    SHAP explanations are a regulatory requirement in production fraud
    detection. This endpoint returns the same prediction as /predict
    plus the features most responsible for the score.
    """
    bundle = request.app.state.bundle
    if bundle.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    explainer = request.app.state.explainer
    if explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not loaded")

    features = build_feature_vector(transaction, bundle)
    fraud_proba = float(bundle.model.predict_proba(features)[0, 1])
    is_fraud = fraud_proba >= bundle.threshold

    top_shap = explainer.top_features(features, bundle.feature_cols, n=5)

    if _prom_available:
        PREDICTION_COUNTER.labels(predicted_class="fraud" if is_fraud else "legit").inc()

    transaction_id = str(uuid.uuid4())[:12]
    timestamp = datetime.utcnow().isoformat()

    background_tasks.add_task(
        log_prediction_async,
        {
            "transaction_id": transaction_id,
            "fraud_probability": fraud_proba,
            "is_fraud": is_fraud,
            "threshold": bundle.threshold,
            "model_version": bundle.version,
            "timestamp": timestamp,
        },
    )
    request.app.state.prediction_count += 1

    return ExplainResponse(
        transaction_id=transaction_id,
        fraud_probability=round(fraud_proba, 6),
        is_fraud=is_fraud,
        threshold=bundle.threshold,
        model_version=bundle.version,
        timestamp=timestamp,
        top_features=[FeatureContribution(**f) for f in top_shap],
    )


@router.get("/predictions/recent")
async def recent_predictions(limit: int = 50):
    """Return the most recent N predictions."""
    limit = max(1, min(limit, 1000))
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in await cursor.fetchall()]


@router.get("/predictions/stats")
async def prediction_stats():
    """Return aggregate prediction statistics."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        cursor = await db.execute("""
            SELECT COUNT(*) as total, SUM(is_fraud) as total_fraud,
                   AVG(fraud_probability) as avg_score
            FROM predictions
        """)
        row = await cursor.fetchone()
        if row and row[0]:
            return {
                "total_predictions": row[0],
                "total_fraud": row[1] or 0,
                "fraud_rate": (row[1] or 0) / row[0],
                "avg_score": row[2],
            }
        return {"total_predictions": 0}
