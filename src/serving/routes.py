"""FastAPI route handlers: /predict, /predict/explain, /health, and prediction history."""

import json
import time
import uuid
from datetime import datetime

import aiosqlite
import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from src.config import DB_PATH, FEATURE_COLS, SHADOW_LOG_PATH
from src.features.engineering import compute_features
from src.serving.schemas import (
    ExplainResponse,
    FeatureContribution,
    HealthResponse,
    NetworkFlowRequest,
    PredictionResponse,
)
from src.serving.shadow_mode import log_shadow_prediction, score_shadow_mode

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
        "PR-AUC of the currently loaded champion model",
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


def build_feature_vector(request: NetworkFlowRequest) -> np.ndarray:
    """
    Convert a NetworkFlowRequest into a model-ready feature vector.

    Applies the same compute_features transform as the training and consumer
    paths (single source of truth), guaranteeing no training-serving skew.

    Parameters
    ----------
    request : NetworkFlowRequest

    Returns
    -------
    features : np.ndarray
        Shape (1, n_features), ordered by FEATURE_COLS.
    """
    feature_dict = compute_features(request.model_dump())
    feature_values = [float(feature_dict[col]) for col in FEATURE_COLS]
    return np.array([feature_values])


# ============= Database Logging =============


async def log_prediction_async(record: dict) -> None:
    """Log a prediction record to the SQLite database."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(
            """INSERT INTO predictions
               (event_id, anomaly_score, is_anomaly, threshold, model_version,
                timestamp, features_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                record["event_id"],
                record["anomaly_score"],
                int(record["is_anomaly"]),
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
    flow: NetworkFlowRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    """
    Predict the anomaly probability for a network flow.

    Returns the attack probability and binary decision at the recall-calibrated
    serving threshold.
    """
    bundle = request.app.state.bundle
    if bundle.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = build_feature_vector(flow)
    anomaly_score = float(bundle.model.predict_proba(features)[0, 1])
    is_anomaly = anomaly_score >= bundle.threshold

    if _prom_available:
        PREDICTION_COUNTER.labels(predicted_class="anomaly" if is_anomaly else "benign").inc()

    event_id = flow.flow_id or str(uuid.uuid4())[:12]
    timestamp = datetime.utcnow().isoformat()

    background_tasks.add_task(
        log_prediction_async,
        {
            "event_id": event_id,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "threshold": bundle.threshold,
            "model_version": bundle.version,
            "timestamp": timestamp,
            "features_json": json.dumps(
                {"src_ip": flow.src_ip, "flow_bytes_per_sec": flow.flow_bytes_per_sec}
            ),
        },
    )
    request.app.state.prediction_count += 1

    # shadow mode: score with the challenger too, log only, serve champion only
    if bundle.challenger_model is not None:
        feature_dict = compute_features(flow.model_dump())
        champion_pred, challenger_pred = score_shadow_mode(
            bundle.model, bundle.challenger_model, feature_dict
        )
        background_tasks.add_task(
            log_shadow_prediction,
            str(SHADOW_LOG_PATH),
            champion_pred,
            challenger_pred,
            champion_pred >= bundle.threshold,
            challenger_pred >= bundle.threshold,
        )

    return PredictionResponse(
        event_id=event_id,
        anomaly_score=round(anomaly_score, 6),
        is_anomaly=is_anomaly,
        threshold=bundle.threshold,
        model_version=bundle.version,
        timestamp=timestamp,
    )


@router.post("/predict/explain", response_model=ExplainResponse)
async def predict_explain(
    flow: NetworkFlowRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> ExplainResponse:
    """
    Predict anomaly probability with top-5 SHAP feature contributions.

    SHAP explanations support analyst triage and audit requirements common in
    production network security operations. This endpoint returns the same
    prediction as /predict plus the features most responsible for the score.
    """
    bundle = request.app.state.bundle
    if bundle.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    explainer = request.app.state.explainer
    if explainer is None:
        raise HTTPException(status_code=503, detail="SHAP explainer not loaded")

    features = build_feature_vector(flow)
    anomaly_score = float(bundle.model.predict_proba(features)[0, 1])
    is_anomaly = anomaly_score >= bundle.threshold

    top_shap = explainer.top_features(features, bundle.feature_cols, n=5)

    if _prom_available:
        PREDICTION_COUNTER.labels(predicted_class="anomaly" if is_anomaly else "benign").inc()

    event_id = flow.flow_id or str(uuid.uuid4())[:12]
    timestamp = datetime.utcnow().isoformat()

    background_tasks.add_task(
        log_prediction_async,
        {
            "event_id": event_id,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "threshold": bundle.threshold,
            "model_version": bundle.version,
            "timestamp": timestamp,
        },
    )
    request.app.state.prediction_count += 1

    return ExplainResponse(
        event_id=event_id,
        anomaly_score=round(anomaly_score, 6),
        is_anomaly=is_anomaly,
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
            SELECT COUNT(*) as total, SUM(is_anomaly) as total_anomaly,
                   AVG(anomaly_score) as avg_score
            FROM predictions
        """)
        row = await cursor.fetchone()
        if row and row[0]:
            return {
                "total_predictions": row[0],
                "total_anomaly": row[1] or 0,
                "anomaly_rate": (row[1] or 0) / row[0],
                "avg_score": row[2],
            }
        return {"total_predictions": 0}
