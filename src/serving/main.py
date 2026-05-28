"""FastAPI application: lifespan, Prometheus instrumentation, router mount, DB init."""

import asyncio
import os
import time
from contextlib import asynccontextmanager

import aiosqlite
import mlflow
import mlflow.tracking
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import DB_PATH, settings
from src.serving.explainer import SHAPExplainer
from src.serving.model_loader import load_champion
from src.serving.routes import router


# ============= Database Initialisation =============


async def init_db() -> None:
    """Create all required tables if they do not already exist."""
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                is_fraud INTEGER NOT NULL,
                threshold REAL NOT NULL,
                model_version TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                features_json TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS drift_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                window_start TEXT,
                window_end TEXT,
                drift_share REAL NOT NULL,
                dataset_drift INTEGER NOT NULL,
                n_drifted_features INTEGER NOT NULL,
                n_features INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS healing_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                details TEXT,
                drift_score REAL,
                model_version TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        await db.commit()


# ============= Model Hot-Reload =============


async def hot_reload_loop(app: FastAPI) -> None:
    """Poll MLflow registry every 60 seconds and reload when @champion version changes."""
    while True:
        await asyncio.sleep(60)
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            client = mlflow.tracking.MlflowClient()
            version_info = client.get_model_version_by_alias(
                settings.mlflow_model_name,
                settings.mlflow_champion_alias,
            )
            current_version = float(version_info.version)
            if current_version != app.state.bundle.model_mtime:
                app.state.bundle = load_champion()
                try:
                    app.state.explainer = SHAPExplainer(app.state.bundle.model)
                except Exception:
                    app.state.explainer = None
                print(f"Hot-reloaded model: version {app.state.bundle.version}")
        except Exception as e:
            print(f"Hot-reload check failed (non-fatal): {e}")


# ============= Lifespan =============


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, load model + SHAP, start hot-reload loop."""
    await init_db()

    app.state.bundle = load_champion()
    app.state.start_time = time.time()
    app.state.prediction_count = 0

    if app.state.bundle.model is not None:
        try:
            app.state.explainer = SHAPExplainer(app.state.bundle.model)
            print("SHAP explainer loaded")
        except Exception as e:
            app.state.explainer = None
            print(f"SHAP explainer failed to load (non-fatal): {e}")
    else:
        app.state.explainer = None

    reload_task = asyncio.create_task(hot_reload_loop(app))

    yield

    reload_task.cancel()
    try:
        await reload_task
    except asyncio.CancelledError:
        pass


# ============= Application =============


app = FastAPI(
    title="Real-Time Fraud Detection API",
    description=(
        "Kafka-backed fraud detection with LightGBM, F2-optimized threshold, "
        "SHAP explanations, and self-healing retraining."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS", settings.allowed_origins
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

app.include_router(router)
