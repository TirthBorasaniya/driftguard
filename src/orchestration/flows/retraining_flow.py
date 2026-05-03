"""Prefect retraining flow: 9-step loop from GE validation to champion promotion."""

import json

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger
from prefect.task_runners import SequentialTaskRunner

from src.config import (
    CHAMPION_METRICS_PATH,
    TRAIN_FILE,
    settings,
)


# ============= Tasks =============


@task(name="validate-data")
def validate_data_task() -> tuple[bool, list[str]]:
    """
    Step 1-2: Run Great Expectations suite. Abort flow if validation fails.

    Returns
    -------
    passed : bool
    failures : list of str
    """
    logger = get_run_logger()

    if not TRAIN_FILE.exists():
        logger.error(f"Training data not found: {TRAIN_FILE}")
        return False, ["Training data file missing"]

    df = pd.read_parquet(TRAIN_FILE)
    from src.validation.expectations import run_validation_suite
    passed, failures = run_validation_suite(df)

    if passed:
        logger.info("Data validation passed")
    else:
        logger.error(f"Data validation failed: {failures}")

    return passed, failures


@task(name="materialize-features")
def materialize_features_task() -> None:
    """Step 3: Materialize Feast features to the Redis online store."""
    logger = get_run_logger()
    try:
        from src.features.materializer import materialize_to_online_store
        materialize_to_online_store()
        logger.info("Feature materialization complete")
    except Exception as e:
        logger.warning(f"Feature materialization failed (non-fatal): {e}")


@task(name="load-data")
def load_data_task():
    """Step 4: Load train/test splits from parquet."""
    import numpy as np

    from src.config import CATEGORICAL_COLS, FEATURE_COLS, TARGET_COL, TEST_FILE

    df_train = pd.read_parquet(TRAIN_FILE)
    df_test = pd.read_parquet(TEST_FILE)

    available = [c for c in FEATURE_COLS if c in df_train.columns]

    X_train = df_train[available].values
    y_train = df_train[TARGET_COL].values
    X_test = df_test[available].values
    y_test = df_test[TARGET_COL].values

    get_run_logger().info(f"Loaded train: {X_train.shape}, test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


@task(name="train-challenger", retries=1, retry_delay_seconds=60)
def train_challenger_task(X_train, y_train, X_test, y_test):
    """Step 5: Train a LightGBM challenger model."""
    from src.training.train import train_lgbm, log_to_mlflow
    logger = get_run_logger()

    model = train_lgbm(X_train, y_train, X_test, y_test)
    run_id, model_version = log_to_mlflow(model, {})
    logger.info(f"Challenger trained. MLflow version: {model_version}")
    return model, model_version


@task(name="evaluate-challenger")
def evaluate_challenger_task(model, X_test, y_test):
    """Steps 6-7: Evaluate challenger and compute F2 threshold."""
    import numpy as np
    from src.training.threshold import find_f2_threshold
    from src.training.evaluate import evaluate_model

    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = find_f2_threshold(y_test, y_proba)
    metrics = evaluate_model(model, X_test, y_test, threshold)

    logger = get_run_logger()
    logger.info(
        f"Challenger AUC-PR: {metrics['auc_pr']:.4f}, "
        f"F2: {metrics['f2_score']:.4f}, threshold: {threshold:.4f}"
    )
    return metrics, threshold


@task(name="load-champion-metrics")
def load_champion_metrics_task() -> dict | None:
    """Step 7: Load champion metrics from JSON artifact (no MLflow API call)."""
    if CHAMPION_METRICS_PATH.exists():
        with open(CHAMPION_METRICS_PATH) as f:
            return json.load(f)
    get_run_logger().info("No champion metrics file found — first training run.")
    return None


@task(name="compare-models")
def compare_task(challenger_metrics: dict, champion_metrics: dict | None) -> bool:
    """Step 8: Challenger wins if AUC-PR improvement exceeds threshold."""
    from src.training.train import should_promote
    result = should_promote(challenger_metrics, champion_metrics)
    get_run_logger().info(f"Promotion decision: {result}")
    return result


@task(name="promote-champion")
def promote_champion_task(model_version: int, metrics: dict, threshold: float) -> None:
    """Step 9: Set champion alias and write champion_metrics.json."""
    from src.training.train import promote_champion
    promote_champion(model_version, metrics, threshold)

    # save production artifacts for API hot-reload
    import joblib
    import mlflow
    import mlflow.lightgbm

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{settings.mlflow_model_name}@{settings.mlflow_champion_alias}"
    model = mlflow.lightgbm.load_model(model_uri)

    from src.training.train import save_production_artifacts
    save_production_artifacts(model, threshold)
    get_run_logger().info("Champion promoted and production artifacts updated")


# ============= Flow =============


@flow(name="retraining-flow", task_runner=SequentialTaskRunner())
def retraining_flow() -> bool:
    """
    9-step self-healing retraining loop.

    Step 1-2 : Great Expectations validation (abort on failure)
    Step 3   : Feast feature materialization
    Step 4   : Load training data
    Step 5   : Train LightGBM challenger
    Step 6   : Evaluate challenger (AUC-PR, F2, threshold)
    Step 7   : Load champion metrics from JSON
    Step 8   : Compare challenger vs champion
    Step 9   : Promote challenger if improvement > threshold

    Returns
    -------
    promoted : bool
        True if a new champion was promoted.
    """
    logger = get_run_logger()
    logger.info("Retraining flow started")

    # steps 1-2: validation gate
    passed, failures = validate_data_task()
    if not passed:
        logger.error(f"Aborting: data validation failed. Failures: {failures}")
        return False

    # step 3
    materialize_features_task()

    # step 4
    X_train, y_train, X_test, y_test = load_data_task()

    # step 5
    model, model_version = train_challenger_task(X_train, y_train, X_test, y_test)

    # steps 6-7
    challenger_metrics, threshold = evaluate_challenger_task(model, X_test, y_test)
    champion_metrics = load_champion_metrics_task()

    # step 8
    should_promote = compare_task(challenger_metrics, champion_metrics)

    # step 9
    if should_promote:
        promote_champion_task(model_version, challenger_metrics, threshold)
        logger.info("New champion promoted. API will hot-reload on next poll.")
    else:
        logger.info("Champion retained. Challenger registered but not promoted.")

    return should_promote


@flow(name="materialization-flow")
def materialization_flow() -> None:
    """Standalone flow to refresh Feast online store features."""
    materialize_features_task()


if __name__ == "__main__":
    retraining_flow()
