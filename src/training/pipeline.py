"""Prefect training pipeline: validate, train, compare, promote."""

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger
from prefect.task_runners import SequentialTaskRunner

from src.config import (
    CATEGORICAL_COLS,
    CHAMPION_IMPROVEMENT_THRESHOLD,
    FEATURE_COLS,
    LGBM_PARAMS,
    MLFLOW_CHALLENGER_ALIAS,
    PROCESSED_DIR,
    TARGET_COL,
)
from src.training.train import (
    apply_smote,
    get_champion_metrics,
    load_training_data,
    log_to_mlflow,
    promote_model,
    save_production_artifacts,
    train_model,
)


# ============= Pipeline Tasks =============


@task(name="validate-data")
def validate_data():
    """Validate that processed data files exist and contain expected columns."""
    logger = get_run_logger()

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    df = pd.read_csv(train_path, nrows=5)
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in training data: {missing_cols}")

    logger.info("Data validation passed")
    return True


@task(name="train-model", retries=1, retry_delay_seconds=30)
def train_model_task():
    """Train LightGBM with SMOTE and evaluate on held-out test set."""
    logger = get_run_logger()

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

    logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}, AUC-PR: {metrics['auc_pr']:.4f}")
    return model, metrics


@task(name="register-model")
def register_model_task(model, metrics):
    """Register trained model in MLflow model registry."""
    run_id, model_version = log_to_mlflow(model, metrics, LGBM_PARAMS)
    return model_version, metrics


@task(name="compare-and-promote")
def compare_and_promote_task(model_version, metrics):
    """
    Compare new model against champion and promote if performance exceeds threshold.

    Returns
    -------
    promoted : bool
        True if the new model was promoted to champion.
    """
    logger = get_run_logger()

    champion_metrics = get_champion_metrics()

    if champion_metrics is None:
        logger.info("No existing champion. Promoting new model.")
        promote_model(model_version)
        return True

    current_auc_pr = champion_metrics.get("auc_pr", 0)
    new_auc_pr = metrics["auc_pr"]
    improvement = new_auc_pr - current_auc_pr

    if improvement > CHAMPION_IMPROVEMENT_THRESHOLD:
        logger.info(
            f"Promoting: new AUC-PR {new_auc_pr:.4f} vs champion {current_auc_pr:.4f} "
            f"(+{improvement:.4f})"
        )
        promote_model(model_version)
        return True

    logger.info(
        f"Keeping champion: improvement {improvement:.4f} below "
        f"threshold {CHAMPION_IMPROVEMENT_THRESHOLD}"
    )
    promote_model(model_version, alias=MLFLOW_CHALLENGER_ALIAS)
    return False


@task(name="save-artifacts")
def save_artifacts_task(model, metrics):
    """Save production model and metadata artifacts to disk."""
    save_production_artifacts(model, metrics)


# ============= Pipeline Flow =============


@flow(name="training-pipeline", task_runner=SequentialTaskRunner())
def training_pipeline():
    """
    Full training pipeline: validate data, train model, compare against
    champion, promote if better, and save production artifacts.
    """
    logger = get_run_logger()
    logger.info("Starting training pipeline")

    validate_data()

    model, metrics = train_model_task()
    model_version, metrics = register_model_task(model, metrics)
    promoted = compare_and_promote_task(model_version, metrics)
    save_artifacts_task(model, metrics)

    logger.info(f"Pipeline complete. Model promoted to champion: {promoted}")
    return promoted


if __name__ == "__main__":
    training_pipeline()
