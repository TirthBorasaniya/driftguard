"""Model training: LightGBM with SMOTE oversampling and MLflow experiment tracking."""

import json
import time

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)

from src.config import (
    CATEGORICAL_COLS,
    CATEGORICAL_COLS_PATH,
    CHAMPION_IMPROVEMENT_THRESHOLD,
    EARLY_STOPPING_ROUNDS,
    FEATURE_COLS,
    FEATURE_COLS_PATH,
    LGBM_PARAMS,
    LOG_EVALUATION_PERIOD,
    METRICS_HISTORY_PATH,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_CHALLENGER_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    NUMERIC_COLS,
    PROCESSED_DIR,
    PRODUCTION_MODEL_PATH,
    SMOTE_RANDOM_STATE,
    TARGET_COL,
)


# ============= Data Loading =============


def load_training_data():
    """
    Load train and test splits from processed directory.

    Returns
    -------
    df_train : pd.DataFrame
        Training data.
    df_test : pd.DataFrame
        Test data for evaluation.
    """
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Train: {df_train.shape}, Test: {df_test.shape}")
    print(f"Train fraud rate: {df_train[TARGET_COL].mean():.4f}")
    print(f"Test fraud rate: {df_test[TARGET_COL].mean():.4f}")

    return df_train, df_test


# ============= SMOTE Oversampling =============


def apply_smote(X_train, y_train):
    """
    Apply SMOTE oversampling to balance classes.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    X_resampled : np.ndarray
        Oversampled feature matrix.
    y_resampled : np.ndarray
        Oversampled labels.
    """
    print(f"Before SMOTE: {np.bincount(y_train.astype(int))}")
    smote = SMOTE(random_state=SMOTE_RANDOM_STATE, n_jobs=-1)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE:  {np.bincount(y_resampled.astype(int))}")
    return X_resampled, y_resampled


# ============= Model Training =============


def train_model(X_train, y_train, X_test, y_test, cat_indices):
    """
    Train LightGBM classifier with early stopping.

    Parameters
    ----------
    X_train : np.ndarray
        Training features (post-SMOTE).
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Held-out test features.
    y_test : np.ndarray
        Held-out test labels.
    cat_indices : list of int
        Indices of categorical features for native LightGBM handling.

    Returns
    -------
    model : lgb.LGBMClassifier
        Trained model.
    metrics : dict
        Evaluation metrics on test set.
    """
    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=["auc", "binary_logloss"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=LOG_EVALUATION_PERIOD),
        ],
        categorical_feature=cat_indices if cat_indices else "auto",
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, y_pred_proba)),
        "auc_pr": float(average_precision_score(y_test, y_pred_proba)),
        "best_iteration": int(model.best_iteration_),
        "n_estimators_used": int(model.best_iteration_),
    }

    print(f"\nEvaluation Metrics:")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:  {metrics['auc_pr']:.4f}")
    print(f"  Best iteration: {metrics['best_iteration']}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'])}")

    return model, metrics


# ============= MLflow Integration =============


def log_to_mlflow(model, metrics, params):
    """
    Log model, metrics, and parameters to MLflow.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Trained model.
    metrics : dict
        Evaluation metrics.
    params : dict
        Hyperparameters.

    Returns
    -------
    run_id : str
        MLflow run ID.
    model_version : int
        Registered model version number.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # log only serializable params
        safe_params = {k: str(v) if isinstance(v, list) else v for k, v in params.items()}
        mlflow.log_params(safe_params)
        mlflow.log_metrics(metrics)

        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
    latest_version = max(int(v.version) for v in versions)

    return run_id, latest_version


def promote_model(model_version, alias=None):
    """
    Set an alias on a model version in the MLflow registry.

    Parameters
    ----------
    model_version : int
        Model version to promote.
    alias : str or None
        Alias to set. Defaults to champion.
    """
    if alias is None:
        alias = MLFLOW_CHAMPION_ALIAS

    client = mlflow.tracking.MlflowClient()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client.set_registered_model_alias(
        name=MLFLOW_MODEL_NAME,
        alias=alias,
        version=model_version,
    )
    print(f"Promoted version {model_version} to alias '{alias}'")


def get_champion_metrics():
    """
    Retrieve the current champion model's metrics from MLflow.

    Returns
    -------
    metrics : dict or None
        Champion metrics, or None if no champion exists.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        version_info = client.get_model_version_by_alias(
            name=MLFLOW_MODEL_NAME,
            alias=MLFLOW_CHAMPION_ALIAS,
        )
        run = client.get_run(version_info.run_id)
        return run.data.metrics
    except Exception:
        return None


def get_champion_version():
    """
    Get the current champion model version number.

    Returns
    -------
    version : int or None
        Champion version, or None if no champion exists.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        version_info = client.get_model_version_by_alias(
            name=MLFLOW_MODEL_NAME,
            alias=MLFLOW_CHAMPION_ALIAS,
        )
        return int(version_info.version)
    except Exception:
        return None


# ============= Artifact Persistence =============


def save_production_artifacts(model, metrics):
    """
    Save model and metadata to disk for deployment.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Trained model.
    metrics : dict
        Evaluation metrics.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, PRODUCTION_MODEL_PATH)
    print(f"Saved model: {PRODUCTION_MODEL_PATH}")

    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    with open(CATEGORICAL_COLS_PATH, "w") as f:
        json.dump(CATEGORICAL_COLS, f, indent=2)

    # append to metrics history
    metrics_row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now().isoformat(),
        **metrics,
    }])
    if METRICS_HISTORY_PATH.exists():
        existing = pd.read_csv(METRICS_HISTORY_PATH)
        metrics_row = pd.concat([existing, metrics_row], ignore_index=True)
    metrics_row.to_csv(METRICS_HISTORY_PATH, index=False)
    print(f"Saved metrics history: {METRICS_HISTORY_PATH}")


# ============= Full Training Run =============


def run_training():
    """Execute the full training pipeline: load, SMOTE, train, log, promote, save."""
    print("=" * 60)
    print("DriftGuard: Model Training")
    print("=" * 60)

    start_time = time.time()

    df_train, df_test = load_training_data()

    available_features = [c for c in FEATURE_COLS if c in df_train.columns]
    cat_indices = [
        available_features.index(c)
        for c in CATEGORICAL_COLS
        if c in available_features
    ]

    X_train = df_train[available_features].values
    y_train = df_train[TARGET_COL].values
    X_test = df_test[available_features].values
    y_test = df_test[TARGET_COL].values

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    model, metrics = train_model(X_train_smote, y_train_smote, X_test, y_test, cat_indices)

    run_id, model_version = log_to_mlflow(model, metrics, LGBM_PARAMS)

    champion_metrics = get_champion_metrics()
    if champion_metrics is None:
        print("No existing champion. Promoting new model.")
        promote_model(model_version)
    elif metrics["auc_pr"] > champion_metrics.get("auc_pr", 0) + CHAMPION_IMPROVEMENT_THRESHOLD:
        print(
            f"New AUC-PR ({metrics['auc_pr']:.4f}) exceeds champion "
            f"({champion_metrics.get('auc_pr', 0):.4f}). Promoting."
        )
        promote_model(model_version)
    else:
        print(
            f"New AUC-PR ({metrics['auc_pr']:.4f}) does not exceed champion. "
            "Registering as challenger."
        )
        promote_model(model_version, alias=MLFLOW_CHALLENGER_ALIAS)

    save_production_artifacts(model, metrics)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed / 60:.1f} minutes")

    return model, metrics


if __name__ == "__main__":
    run_training()
