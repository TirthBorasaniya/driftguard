"""Prefect self-healing flow: detect drift, decide action, retrain, validate, promote."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from prefect import flow, task
from prefect.logging import get_run_logger
from prefect.task_runners import SequentialTaskRunner

from src.config import (
    BATCH_DIR,
    BATCH_FILES,
    DB_PATH,
    DRIFT_SHARE_THRESHOLD,
    HEALING_COOLDOWN_HOURS,
    HEALING_MODE,
    REFERENCE_FILE,
)
from src.monitoring.drift_detector import log_drift_report, run_drift_report


# ============= Healing Event Logging =============


def log_healing_event(event_type, details, drift_score=None, model_version=None):
    """
    Persist a self-healing event to the database.

    Parameters
    ----------
    event_type : str
        One of: DRIFT_DETECTED, RETRAIN_TRIGGERED, MODEL_PROMOTED,
        CHALLENGER_REGISTERED, ALERT_SENT, COOLDOWN_ACTIVE.
    details : str
        Human-readable description of the event.
    drift_score : float or None
        Associated drift share score.
    model_version : str or None
        Model version involved.
    """
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """INSERT INTO healing_events
               (event_type, details, timestamp, drift_score, model_version, healing_mode)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                event_type,
                details,
                datetime.now().isoformat(),
                drift_score,
                model_version,
                HEALING_MODE,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_last_healing_time():
    """
    Get the timestamp of the most recent retraining event.

    Returns
    -------
    last_time : datetime or None
        Timestamp of last RETRAIN_TRIGGERED event.
    """
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cursor = conn.execute(
            "SELECT timestamp FROM healing_events WHERE event_type='RETRAIN_TRIGGERED' "
            "ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            return datetime.fromisoformat(row[0])
        return None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


# ============= Pipeline Tasks =============


@task(name="check-drift")
def check_drift_task(batch_path=None):
    """
    Run drift detection on the latest production batch.

    Parameters
    ----------
    batch_path : str or None
        Path to specific batch CSV. If None, uses the latest batch file.

    Returns
    -------
    result : dict
        Drift detection result.
    """
    logger = get_run_logger()

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Reference data not found: {REFERENCE_FILE}")

    reference_df = pd.read_csv(REFERENCE_FILE)

    if batch_path is not None:
        current_path = Path(batch_path)
    else:
        # find the latest batch file
        batch_files = sorted(BATCH_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        if not batch_files:
            # fall back to predefined batch files
            batch_files = [f for f in BATCH_FILES if f.exists()]
        if not batch_files:
            raise FileNotFoundError("No batch files found for drift detection")
        current_path = batch_files[-1]

    current_df = pd.read_csv(current_path)
    batch_name = current_path.stem

    logger.info(f"Checking drift: {batch_name} ({len(current_df)} rows)")

    result, report_path = run_drift_report(reference_df, current_df, batch_name)
    log_drift_report(result)

    return result


@task(name="check-cooldown")
def check_cooldown_task():
    """
    Verify the cooldown period has elapsed since last retraining.

    Returns
    -------
    can_retrain : bool
        True if cooldown has expired.
    """
    logger = get_run_logger()

    last_time = get_last_healing_time()
    if last_time is None:
        logger.info("No previous retraining events. Cooldown clear.")
        return True

    cooldown_end = last_time + timedelta(hours=HEALING_COOLDOWN_HOURS)
    now = datetime.now()

    if now >= cooldown_end:
        logger.info(f"Cooldown expired. Last retrain: {last_time.isoformat()}")
        return True

    remaining = cooldown_end - now
    logger.info(
        f"Cooldown active. {remaining.total_seconds() / 3600:.1f} hours remaining. "
        f"Last retrain: {last_time.isoformat()}"
    )
    log_healing_event(
        "COOLDOWN_ACTIVE",
        f"Cooldown prevents retraining for {remaining.total_seconds() / 3600:.1f} more hours",
    )
    return False


@task(name="trigger-retraining")
def trigger_retraining_task(drift_result):
    """
    Execute model retraining via the training pipeline.

    Parameters
    ----------
    drift_result : dict
        Drift detection result for logging context.

    Returns
    -------
    promoted : bool
        True if the new model was promoted to champion.
    """
    logger = get_run_logger()

    log_healing_event(
        "RETRAIN_TRIGGERED",
        f"Drift share: {drift_result['drift_share']:.2%}, "
        f"drifted features: {drift_result['n_drifted_features']}",
        drift_score=drift_result["drift_share"],
    )

    from src.training.train import run_training

    model, metrics = run_training()

    logger.info(f"Retraining complete. AUC-PR: {metrics['auc_pr']:.4f}")

    log_healing_event(
        "MODEL_PROMOTED" if HEALING_MODE == "AUTO" else "CHALLENGER_REGISTERED",
        f"AUC-ROC: {metrics['auc_roc']:.4f}, AUC-PR: {metrics['auc_pr']:.4f}",
        model_version=f"retrain-{datetime.now().strftime('%Y%m%d%H%M%S')}",
    )

    return True


@task(name="send-alert")
def send_alert_task(drift_result):
    """
    Log a drift alert without triggering retraining.

    Parameters
    ----------
    drift_result : dict
        Drift detection result.
    """
    logger = get_run_logger()

    drifted_cols = [d["column"] for d in drift_result.get("drifted_columns", [])]
    detail = (
        f"Drift detected: {drift_result['drift_share']:.2%} features drifted. "
        f"Columns: {', '.join(drifted_cols[:10])}"
    )

    logger.warning(detail)
    log_healing_event(
        "ALERT_SENT",
        detail,
        drift_score=drift_result["drift_share"],
    )


# ============= Self-Healing Flow =============


@flow(name="self-healing-pipeline", task_runner=SequentialTaskRunner())
def self_healing_pipeline(batch_path=None):
    """
    Self-healing pipeline: detect drift, check cooldown, decide action based
    on HEALING_MODE configuration.

    Modes
    -----
    AUTO : Retrain and promote automatically if drift is detected.
    SHADOW : Retrain but register as challenger for manual review.
    ALERT_ONLY : Log alert without retraining.

    Parameters
    ----------
    batch_path : str or None
        Path to specific batch file. If None, uses the latest.
    """
    logger = get_run_logger()
    logger.info(f"Self-healing pipeline started. Mode: {HEALING_MODE}")

    # step 1: check drift
    drift_result = check_drift_task(batch_path)

    # step 2: evaluate drift severity
    drift_share = drift_result.get("drift_share", 0)
    dataset_drift = drift_result.get("dataset_drift", False)

    if not dataset_drift and drift_share < DRIFT_SHARE_THRESHOLD:
        logger.info(
            f"No significant drift detected (share: {drift_share:.2%}). "
            "No action required."
        )
        return False

    logger.warning(
        f"Drift detected: share={drift_share:.2%}, "
        f"dataset_drift={dataset_drift}"
    )
    log_healing_event(
        "DRIFT_DETECTED",
        f"Drift share: {drift_share:.2%}, batch: {drift_result['batch_name']}",
        drift_score=drift_share,
    )

    # step 3: decide action based on mode
    if HEALING_MODE == "ALERT_ONLY":
        send_alert_task(drift_result)
        return False

    # step 4: check cooldown
    can_retrain = check_cooldown_task()
    if not can_retrain:
        logger.info("Cooldown active. Skipping retraining.")
        return False

    # step 5: retrain
    if HEALING_MODE in ("AUTO", "SHADOW"):
        trigger_retraining_task(drift_result)
        return True

    return False


# ============= Main =============


def run_self_healing():
    """Execute the self-healing pipeline on all available batches."""
    print("=" * 60)
    print(f"DriftGuard: Self-Healing Pipeline (mode: {HEALING_MODE})")
    print("=" * 60)

    batch_files = sorted(BATCH_DIR.glob("*.csv"))
    if not batch_files:
        batch_files = [f for f in BATCH_FILES if f.exists()]

    if not batch_files:
        print("No batch files found. Run preprocessing first.")
        return

    for batch_path in batch_files:
        print(f"\nProcessing: {batch_path.name}")
        try:
            self_healing_pipeline(str(batch_path))
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    run_self_healing()
