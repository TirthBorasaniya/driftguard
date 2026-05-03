"""Model loading: champion model by MLflow alias, threshold JSON, and SafeLabelEncoders."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.lightgbm

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
    model_mtime: float


def load_champion() -> ModelBundle:
    """
    Load the champion model, F2 threshold, and encoders from disk.

    Tries MLflow alias first, falls back to local production_model.pkl.
    Threshold is always loaded from threshold.json artifact.

    Returns
    -------
    bundle : ModelBundle
    """
    model = None
    version = "none"
    model_mtime = 0.0

    if PRODUCTION_MODEL_PATH.exists():
        model = joblib.load(PRODUCTION_MODEL_PATH)
        model_mtime = PRODUCTION_MODEL_PATH.stat().st_mtime
        from datetime import datetime
        version = datetime.fromtimestamp(model_mtime).strftime("%Y%m%d-%H%M%S")
        print(f"Loaded model: {PRODUCTION_MODEL_PATH.name} (version {version})")
    else:
        print(f"WARNING: model not found at {PRODUCTION_MODEL_PATH}")

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
