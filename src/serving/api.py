"""FastAPI serving layer with prediction logging, Prometheus metrics, and model hot-reload."""

import asyncio
import json
import os
import pickle
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite
import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from src.config import (
    API_HOST,
    API_PORT,
    CATEGORICAL_COLS,
    D_COLS_TO_NORMALIZE,
    DB_PATH,
    ENCODERS_DIR,
    FEATURE_COLS,
    METRICS_ENABLED,
    MODELS_DIR,
    NUMERIC_COLS,
    PRODUCTION_MODEL_PATH,
    TARGET_COL,
)


# ============= Request/Response Models =============


class PredictionRequest(BaseModel):
    """Transaction features for fraud prediction. Extra fields are accepted."""

    model_config = ConfigDict(extra="allow")

    TransactionAmt: float
    TransactionDT: float


class PredictionResponse(BaseModel):
    """Fraud prediction result."""

    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    """API health status."""

    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    total_predictions: int


# ============= Database =============


async def init_db():
    """Initialize the predictions database with WAL mode."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                is_fraud INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                features_json TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS healing_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                details TEXT,
                timestamp TEXT NOT NULL,
                drift_score REAL,
                model_version TEXT,
                healing_mode TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS drift_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_name TEXT NOT NULL,
                drift_share REAL NOT NULL,
                dataset_drift INTEGER NOT NULL,
                n_drifted_features INTEGER NOT NULL,
                n_features INTEGER NOT NULL,
                report_path TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        await db.commit()


async def log_prediction(prediction_data: dict):
    """Log a prediction to the database in the background."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(
            """INSERT INTO predictions
               (transaction_id, fraud_probability, is_fraud, model_version, timestamp, features_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                prediction_data["transaction_id"],
                prediction_data["fraud_probability"],
                int(prediction_data["is_fraud"]),
                prediction_data["model_version"],
                prediction_data["timestamp"],
                prediction_data.get("features_json", "{}"),
            ),
        )
        await db.commit()


# ============= Model Loading =============


def load_model_artifacts(app_state):
    """
    Load model, encoders, and feature metadata into app state.

    Parameters
    ----------
    app_state : State
        FastAPI app.state object to populate.
    """
    app_state.model = None
    app_state.encoders = {}
    app_state.model_version = "none"
    app_state.model_mtime = 0

    if PRODUCTION_MODEL_PATH.exists():
        app_state.model = joblib.load(PRODUCTION_MODEL_PATH)
        app_state.model_mtime = PRODUCTION_MODEL_PATH.stat().st_mtime
        app_state.model_version = datetime.fromtimestamp(
            app_state.model_mtime
        ).strftime("%Y%m%d-%H%M%S")
        print(f"Loaded model: {PRODUCTION_MODEL_PATH}")
    else:
        print(f"Model not found at {PRODUCTION_MODEL_PATH}, API will return 503 on /predict")

    # load encoders
    if ENCODERS_DIR.exists():
        for col in CATEGORICAL_COLS:
            encoder_path = ENCODERS_DIR / f"{col}_encoder.pkl"
            if encoder_path.exists():
                with open(encoder_path, "rb") as f:
                    app_state.encoders[col] = pickle.load(f)
        print(f"Loaded {len(app_state.encoders)} encoders")


# ============= Feature Preparation =============


def prepare_features(raw: dict, encoders: dict) -> np.ndarray:
    """
    Transform raw transaction features into model input vector.

    Reproduces the exact same transformations as preprocess.py to ensure
    training-serving consistency.

    Parameters
    ----------
    raw : dict
        Raw feature values from the prediction request.
    encoders : dict
        Loaded LabelEncoder objects keyed by column name.

    Returns
    -------
    features : np.ndarray
        Shape (1, n_features) array ready for model.predict_proba.
    """
    # engineered features
    raw["TransactionAmt_log"] = np.log1p(raw.get("TransactionAmt", 0))

    transaction_dt = raw.get("TransactionDT", 0)
    raw["transaction_day"] = (transaction_dt / 86400) % 7
    raw["transaction_hour"] = (transaction_dt / 3600) % 24

    # normalize D columns
    for col in D_COLS_TO_NORMALIZE:
        col_norm = f"{col}n"
        val = raw.get(col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            raw[col_norm] = val - raw["transaction_day"]
        else:
            raw[col_norm] = -999

    # build feature vector in FEATURE_COLS order
    feature_values = []
    for col in FEATURE_COLS:
        if col in CATEGORICAL_COLS:
            val = str(raw.get(col, "unknown") or "unknown")
            encoder = encoders.get(col)
            if encoder is not None:
                known = set(encoder.classes_)
                val = val if val in known else "unknown"
                val = int(encoder.transform([val])[0])
            else:
                val = 0
            feature_values.append(val)
        else:
            val = raw.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = -999.0
            feature_values.append(float(val))

    return np.array([feature_values])


# ============= Model Hot-Reload =============


async def model_hot_reload_loop(app):
    """Background task that checks for model file updates and reloads."""
    while True:
        await asyncio.sleep(60)
        try:
            if PRODUCTION_MODEL_PATH.exists():
                current_mtime = PRODUCTION_MODEL_PATH.stat().st_mtime
                if current_mtime > app.state.model_mtime:
                    load_model_artifacts(app.state)
                    print(f"Model hot-reloaded: version {app.state.model_version}")
        except Exception as e:
            print(f"Hot-reload check failed: {e}")


# ============= Application Lifespan =============


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for the FastAPI application."""
    # startup
    await init_db()
    load_model_artifacts(app.state)
    app.state.start_time = time.time()
    app.state.prediction_count = 0

    # start hot-reload background task
    reload_task = asyncio.create_task(model_hot_reload_loop(app))

    yield

    # shutdown
    reload_task.cancel()
    try:
        await reload_task
    except asyncio.CancelledError:
        pass


# ============= Application =============


app = FastAPI(
    title="DriftGuard Fraud Detection API",
    description="Real-time fraud prediction with drift monitoring and self-healing retraining.",
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Prometheus Instrumentation =============

if METRICS_ENABLED:
    try:
        from prometheus_client import Counter, Gauge, Histogram
        from prometheus_fastapi_instrumentator import Instrumentator

        PREDICTION_COUNTER = Counter(
            "model_predictions_total",
            "Total predictions by class",
            ["predicted_class"],
        )
        PREDICTION_LATENCY = Histogram(
            "model_prediction_latency_seconds",
            "Time spent computing prediction",
        )
        PREDICTION_SCORE = Histogram(
            "model_prediction_score",
            "Distribution of fraud probability scores",
            buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        )
        DRIFT_SCORE_GAUGE = Gauge(
            "model_drift_score",
            "Latest drift score from batch monitoring",
        )
        MODEL_VERSION_GAUGE = Gauge(
            "model_version_timestamp",
            "Timestamp of currently loaded model",
        )

        Instrumentator().instrument(app).expose(app)
        _prometheus_available = True
    except ImportError:
        _prometheus_available = False
else:
    _prometheus_available = False


# ============= Endpoints =============


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint with model and system status."""
    return HealthResponse(
        status="healthy",
        model_loaded=app.state.model is not None,
        model_version=app.state.model_version,
        uptime_seconds=time.time() - app.state.start_time,
        total_predictions=app.state.prediction_count,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Predict fraud probability for a transaction.

    Accepts TransactionAmt and TransactionDT as required fields, plus any
    additional feature fields. Missing features are imputed with defaults.
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    raw_features = request.model_dump()
    features = prepare_features(raw_features, app.state.encoders)

    fraud_proba = float(app.state.model.predict_proba(features)[0, 1])
    is_fraud = fraud_proba >= 0.5

    latency = time.time() - start

    # update prometheus metrics
    if _prometheus_available:
        PREDICTION_COUNTER.labels(predicted_class="fraud" if is_fraud else "legit").inc()
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_SCORE.observe(fraud_proba)
        MODEL_VERSION_GAUGE.set(app.state.model_mtime)

    transaction_id = str(uuid.uuid4())[:12]
    timestamp = datetime.utcnow().isoformat()

    prediction_data = {
        "transaction_id": transaction_id,
        "fraud_probability": fraud_proba,
        "is_fraud": is_fraud,
        "model_version": app.state.model_version,
        "timestamp": timestamp,
        "features_json": json.dumps({
            "TransactionAmt": raw_features.get("TransactionAmt"),
            "TransactionDT": raw_features.get("TransactionDT"),
        }),
    }

    # log asynchronously to avoid blocking the response
    background_tasks.add_task(log_prediction, prediction_data)
    app.state.prediction_count += 1

    return PredictionResponse(
        transaction_id=transaction_id,
        fraud_probability=round(fraud_proba, 6),
        is_fraud=is_fraud,
        model_version=app.state.model_version,
        timestamp=timestamp,
    )


@app.get("/predictions/recent")
async def recent_predictions(limit: int = 50):
    """Retrieve the most recent predictions."""
    limit = max(1, min(limit, 1000))
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


@app.get("/predictions/stats")
async def prediction_stats():
    """Aggregate prediction statistics."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        cursor = await db.execute("""
            SELECT
                COUNT(*) as total,
                SUM(is_fraud) as total_fraud,
                AVG(fraud_probability) as avg_score,
                MIN(timestamp) as first_prediction,
                MAX(timestamp) as last_prediction
            FROM predictions
        """)
        row = await cursor.fetchone()
        if row and row[0] > 0:
            return {
                "total_predictions": row[0],
                "total_fraud": row[1],
                "fraud_rate": row[1] / row[0] if row[0] > 0 else 0,
                "avg_score": row[2],
                "first_prediction": row[3],
                "last_prediction": row[4],
            }
        return {"total_predictions": 0}


@app.get("/drift/latest")
async def latest_drift():
    """Retrieve the most recent drift report summary."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM drift_reports ORDER BY id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        if row:
            return dict(row)
        return {"message": "No drift reports available"}


@app.get("/drift/history")
async def drift_history(limit: int = 20):
    """Retrieve drift report history."""
    limit = max(1, min(limit, 500))
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM drift_reports ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


@app.get("/healing/events")
async def healing_events(limit: int = 20):
    """Retrieve self-healing event history."""
    limit = max(1, min(limit, 500))
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM healing_events ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


@app.post("/drift/score")
async def update_drift_score(score: float):
    """Update the current drift score (called by drift detector)."""
    if _prometheus_available:
        DRIFT_SCORE_GAUGE.set(score)
    return {"status": "updated", "drift_score": score}
