"""Alert handler: log drift alerts and trigger Prefect retraining flow on breach."""

import sqlite3
from datetime import datetime

from src.config import DB_PATH, settings


def _log_healing_event(event_type: str, details: str, drift_score: float = None) -> None:
    """Persist a healing event to the database."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """INSERT INTO healing_events (event_type, details, drift_score, timestamp)
               VALUES (?, ?, ?, ?)""",
            (event_type, details, drift_score, datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def handle_drift_alert(drift_result: dict) -> None:
    """
    Handle a detected drift event according to HEALING_MODE setting.

    Modes
    -----
    AUTO        : Log alert and immediately trigger Prefect retraining flow.
    SHADOW      : Log alert and trigger retraining, but register as challenger only.
    ALERT_ONLY  : Log alert, no retraining.

    Parameters
    ----------
    drift_result : dict
        Drift report result from run_evidently_report.
    """
    drift_share = drift_result.get("drift_share", 0.0)
    details = (
        f"Drift share: {drift_share:.2%}, "
        f"drifted features: {drift_result.get('n_drifted_features', 0)}"
    )

    _log_healing_event("DRIFT_DETECTED", details, drift_score=drift_share)
    print(f"Alert: {details}")

    if settings.healing_mode == "ALERT_ONLY":
        print("HEALING_MODE=ALERT_ONLY: no retraining triggered.")
        return

    if settings.healing_mode in ("AUTO", "SHADOW"):
        print(f"HEALING_MODE={settings.healing_mode}: triggering retraining flow.")
        _log_healing_event("RETRAIN_TRIGGERED", details, drift_score=drift_share)
        try:
            from src.orchestration.flows.retraining_flow import retraining_flow
            retraining_flow()
        except Exception as e:
            print(f"Retraining flow failed: {e}")
            _log_healing_event("RETRAIN_FAILED", str(e), drift_score=drift_share)
