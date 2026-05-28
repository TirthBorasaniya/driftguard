"""Model loading: champion model by MLflow alias, threshold JSON, and SafeLabelEncoders."""

import json
from dataclasses import dataclass

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.tracking

from src.config import (
    CATEGORICAL_COLS,
    ENCODERS_DIR,
    FEATURE_COLS,
    PRODUCTION_MODEL_PATH,
    THRESHOLD_PATH,
    settings,
)
from src.data.encoders import SafeLabelEncoder, load_encoders


@dataclass
class ModelBundle:
    """All artifacts needed for inference."""

    model: object
    threshold: float
    encoders: dict[str, SafeLabelEncoder]
    feature_cols: list[str]
    version: str
    model_mtime: float  # MLflow version number (float) or file mtime for fallback


def load_champion() -> ModelBundle:
    """
    Load the champion model, F2 threshold, and encoders.

    Primary path: MLflow model registry via @champion alias.
    Fallback path: local production_model.pkl if MLflow registry is unavailable.
    Threshold is always loaded from threshold.json artifact on disk.

    Returns
    -------
    bundle : ModelBundle
    """
    model = None
    version = "none"
    model_mtime = 0.0

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # Primary: load via @champion alias from MLflow model registry
    try:
        model_uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_champion_alias}"
        model = mlflow.lightgbm.load_model(model_uri)
        client = mlflow.tracking.MlflowClient()
        version_info = client.get_model_version_by_alias(
            settings.mlflow_model_name,
            settings.mlflow_champion_alias,
        )
        version = version_info.version
        model_mtime = float(version)
        print(f"Loaded champion from MLflow registry: version {version}")
    except Exception as mlflow_err:
        print(f"MLflow load failed ({mlflow_err}), falling back to local model")
        # Fallback: local joblib written by training pipeline
        if PRODUCTION_MODEL_PATH.exists():
            from datetime import datetime

            model = joblib.load(PRODUCTION_MODEL_PATH)
            model_mtime = PRODUCTION_MODEL_PATH.stat().st_mtime
            version = datetime.fromtimestamp(model_mtime).strftime("%Y%m%d-%H%M%S")
            print(f"Loaded fallback model: {PRODUCTION_MODEL_PATH.name} (version {version})")
        else:
            print("WARNING: no model available — MLflow registry and local disk both failed")

    threshold = 0.5
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            threshold = float(json.load(f)["threshold"])
        print(f"Loaded threshold: {threshold:.4f}")
    else:
        print(f"WARNING: threshold.json not found, using default {threshold}")

    encoders = {}
    if ENCODERS_DIR.exists():
        try:
            encoders = load_encoders(CATEGORICAL_COLS, ENCODERS_DIR)
            print(f"Loaded {len(encoders)} encoders")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    return ModelBundle(
        model=model,
        threshold=threshold,
        encoders=encoders,
        feature_cols=FEATURE_COLS,
        version=version,
        model_mtime=model_mtime,
    )
