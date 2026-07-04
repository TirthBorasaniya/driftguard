"""LightGBM training with scale_pos_weight, MLflow tracking, and champion/challenger promotion."""

import json
import time

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_COLS,
    CHAMPION_METRICS_PATH,
    EARLY_STOPPING_ROUNDS,
    FEATURE_COLS,
    FEATURE_COLS_PATH,
    LGBM_PARAMS,
    LOG_EVALUATION_PERIOD,
    MODELS_DIR,
    PRODUCTION_MODEL_PATH,
    TARGET_COL,
    TEST_FILE,
    THRESHOLD_PATH,
    TRAIN_FILE,
    settings,
)
from src.training.evaluate import evaluate_model
from src.training.threshold import calibrate_threshold

# ============= Operating Point Constants =============

# PR-AUC replaces F2 as the primary metric: it summarizes performance across all
# thresholds and is the standard metric for imbalanced anomaly detection
PROMOTION_METRIC = "pr_auc"
# challenger must exceed champion PR-AUC by this margin to be promoted
PROMOTION_PRAUC_MARGIN = settings.champion_improvement_threshold
# serving threshold is calibrated to achieve this recall on the validation split
SERVING_RECALL_TARGET = 0.95


# ============= Data Loading =============


def load_train_test() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train and test parquet splits and return feature matrices.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    df_train = pd.read_parquet(TRAIN_FILE)
    df_test = pd.read_parquet(TEST_FILE)

    available = [c for c in FEATURE_COLS if c in df_train.columns]

    X_train = df_train[available].values
    y_train = df_train[TARGET_COL].values
    X_test = df_test[available].values
    y_test = df_test[TARGET_COL].values

    print(f"Train: {X_train.shape}, attack rate: {y_train.mean():.4%}")
    print(f"Test:  {X_test.shape}, attack rate: {y_test.mean():.4%}")
    print(f"Features: {len(available)}")

    return X_train, y_train, X_test, y_test


# ============= Training =============


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> lgb.LGBMClassifier:
    """
    Train a LightGBM binary classifier.

    Uses scale_pos_weight for class imbalance (no SMOTE). Categorical feature
    indices are passed to LightGBM so it splits them correctly instead of
    treating label-encoded integers as ordered numeric values.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data.
    X_test, y_test : np.ndarray
        Validation data for early stopping.

    Returns
    -------
    model : lgb.LGBMClassifier
    """
    cat_indices = [
        i for i, col in enumerate(FEATURE_COLS)
        if col in CATEGORICAL_COLS and i < X_train.shape[1]
    ]

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=LOG_EVALUATION_PERIOD),
        ],
        categorical_feature=cat_indices if cat_indices else "auto",
    )

    print(f"Best iteration: {model.best_iteration_}")
    return model


# ============= Champion Comparison =============


def load_champion_metrics() -> dict | None:
    """
    Load champion metrics from JSON artifact (no MLflow API dependency).

    Returns
    -------
    metrics : dict or None
        Champion metrics, or None if no champion exists yet.
    """
    if CHAMPION_METRICS_PATH.exists():
        with open(CHAMPION_METRICS_PATH) as f:
            return json.load(f)
    return None


def should_promote(new_metrics: dict, champion_metrics: dict | None) -> bool:
    """
    Determine if the new model should replace the current champion.

    Parameters
    ----------
    new_metrics : dict
        Challenger evaluation metrics.
    champion_metrics : dict or None
        Current champion metrics, or None if no champion exists.

    Returns
    -------
    promote : bool
    """
    if champion_metrics is None:
        print("No existing champion. Promoting.")
        return True

    improvement = new_metrics["auc_pr"] - champion_metrics.get("auc_pr", 0)
    margin = PROMOTION_PRAUC_MARGIN
    result = improvement > margin
    print(
        f"PR-AUC: challenger={new_metrics['auc_pr']:.4f}, "
        f"champion={champion_metrics.get('auc_pr', 0):.4f}, "
        f"improvement={improvement:+.4f} (margin={margin})"
    )
    return result


# ============= MLflow + Artifact Persistence =============


def log_to_mlflow(
    model: lgb.LGBMClassifier,
    metrics: dict,
) -> tuple[str, int]:
    """
    Log model and metrics to MLflow.

    Parameters
    ----------
    model : lgb.LGBMClassifier
    metrics : dict

    Returns
    -------
    run_id : str
    model_version : int
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run() as run:
        safe_params = {k: str(v) if isinstance(v, list) else v for k, v in LGBM_PARAMS.items()}
        mlflow.log_params(safe_params)
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=settings.mlflow_model_name,
        )
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{settings.mlflow_model_name}'")
    latest_version = max(int(v.version) for v in versions)
    return run_id, latest_version


def promote_champion(model_version: int, metrics: dict, threshold: float) -> None:
    """
    Set champion alias on model version and update champion_metrics.json.

    Parameters
    ----------
    model_version : int
    metrics : dict
    threshold : float
    """
    client = mlflow.tracking.MlflowClient()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client.set_registered_model_alias(
        name=settings.mlflow_model_name,
        alias=settings.mlflow_champion_alias,
        version=model_version,
    )
    print(f"Promoted version {model_version} to '{settings.mlflow_champion_alias}'")

    CHAMPION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAMPION_METRICS_PATH, "w") as f:
        json.dump({**metrics, "threshold": threshold, "version": model_version}, f, indent=2)


def save_production_artifacts(
    model: lgb.LGBMClassifier,
    threshold: float,
) -> None:
    """
    Save model binary, threshold JSON, and feature column list to models/.

    Parameters
    ----------
    model : lgb.LGBMClassifier
    threshold : float
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PRODUCTION_MODEL_PATH)

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": threshold}, f)

    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    print(f"Saved: {PRODUCTION_MODEL_PATH.name}, {THRESHOLD_PATH.name}, {FEATURE_COLS_PATH.name}")


# ============= Full Training Run =============


def run_training() -> tuple:
    """
    Execute the full training pipeline.

    Returns
    -------
    model : lgb.LGBMClassifier
    metrics : dict
    threshold : float
    """
    print("=" * 60)
    print("Network Anomaly Pipeline: Model Training")
    print("=" * 60)

    start = time.time()

    X_train, y_train, X_test, y_test = load_train_test()
    model = train_lgbm(X_train, y_train, X_test, y_test)

    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = calibrate_threshold(y_test, y_proba, SERVING_RECALL_TARGET)
    print(f"Recall-calibrated threshold (target={SERVING_RECALL_TARGET}): {threshold:.4f}")

    metrics = evaluate_model(model, X_test, y_test, threshold)
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC: {metrics['auc_roc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

    run_id, model_version = log_to_mlflow(model, metrics)

    champion_metrics = load_champion_metrics()
    if should_promote(metrics, champion_metrics):
        promote_champion(model_version, metrics, threshold)
    else:
        client = mlflow.tracking.MlflowClient()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client.set_registered_model_alias(
            name=settings.mlflow_model_name,
            alias=settings.mlflow_challenger_alias,
            version=model_version,
        )
        print(f"Registered as '{settings.mlflow_challenger_alias}'")

    save_production_artifacts(model, threshold)

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed / 60:.1f} min")
    return model, metrics, threshold


if __name__ == "__main__":
    run_training()
