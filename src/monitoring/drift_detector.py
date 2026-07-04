"""Streaming drift detector: accumulates events, runs Evidently report per window."""

import sqlite3
from datetime import datetime

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import DataDriftTable, DatasetDriftMetric
from evidently.report import Report

from src.config import (
    DB_PATH,
    REFERENCE_FILE,
    REPORTS_DIR,
    settings,
)
from src.features.engineering import FEATURE_COLS

# reference distribution for PSI/drift computation (benign-only baseline)
REFERENCE_DATA_PATH = str(REFERENCE_FILE)


# ============= Column Mapping =============


def get_column_mapping() -> ColumnMapping:
    """Build Evidently ColumnMapping for the network flow feature set (all numerical)."""
    return ColumnMapping(
        numerical_features=list(FEATURE_COLS),
        categorical_features=[],
    )


# ============= Drift Report =============


def run_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    window_id: str,
) -> dict:
    """
    Run an Evidently data drift report.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Baseline reference dataset.
    current_df : pd.DataFrame
        Current window of production events.
    window_id : str
        Identifier for this window (used in report filename).

    Returns
    -------
    result : dict
        drift_share, dataset_drift flag, n_drifted_features, report_path.
    """
    common_cols = [
        c for c in FEATURE_COLS
        if c in reference_df.columns and c in current_df.columns
    ]
    ref = reference_df[common_cols].copy()
    cur = current_df[common_cols].copy()

    report = Report(metrics=[DatasetDriftMetric(), DataDriftTable()])
    report.run(
        reference_data=ref,
        current_data=cur,
        column_mapping=get_column_mapping(),
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"drift_{window_id}_{ts}.html"
    report.save_html(str(report_path))

    report_dict = report.as_dict()
    result = _parse_report(report_dict)
    result["report_path"] = str(report_path)
    result["window_id"] = window_id

    return result


def _parse_report(report_dict: dict) -> dict:
    """Extract scalar metrics from Evidently report dict."""
    dataset_drift = False
    drift_share = 0.0
    n_drifted = 0
    n_features = 0

    for metric in report_dict.get("metrics", []):
        metric_id = metric.get("metric", "")
        res = metric.get("result", {})
        if metric_id == "DatasetDriftMetric":
            dataset_drift = res.get("dataset_drift", False)
            drift_share = res.get("share_of_drifted_columns", 0.0)
            n_drifted = res.get("number_of_drifted_columns", 0)
            n_features = res.get("number_of_columns", 0)

    return {
        "dataset_drift": dataset_drift,
        "drift_share": drift_share,
        "n_drifted_features": n_drifted,
        "n_features": n_features,
        "timestamp": datetime.now().isoformat(),
    }


# ============= Database Logging =============


def log_drift_result(result: dict) -> None:
    """Persist drift report summary to predictions.db."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """INSERT INTO drift_reports
               (drift_share, dataset_drift, n_drifted_features, n_features, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (
                result["drift_share"],
                int(result["dataset_drift"]),
                result["n_drifted_features"],
                result["n_features"],
                result["timestamp"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ============= Streaming Detector =============


class StreamingDriftDetector:
    """
    Accumulates incoming events into a window buffer and triggers Evidently
    drift reports once the minimum window size is reached.

    The 500-event minimum guard prevents running Evidently's KS test on
    sample sizes where the test is statistically unreliable.
    """

    def __init__(
        self,
        reference_df: pd.DataFrame | None,
        min_window: int = 500,
    ) -> None:
        self.reference_df = reference_df
        self.min_window = min_window
        self.buffer: list[dict] = []
        self.window_count = 0

    def add_event(self, event: dict) -> bool:
        """
        Add an event to the buffer. Triggers report when buffer reaches min_window.

        Parameters
        ----------
        event : dict
            Processed network flow event with computed features.

        Returns
        -------
        drift_triggered : bool
            True if a drift report was generated and drift was detected.
        """
        self.buffer.append(event)

        if len(self.buffer) < self.min_window:
            return False

        if self.reference_df is None:
            print(f"WARNING: No reference data. Skipping drift report ({len(self.buffer)} events).")
            self.buffer = []
            return False

        self.window_count += 1
        window_id = f"window_{self.window_count:04d}"
        current_df = pd.DataFrame(self.buffer)
        self.buffer = []

        print(f"Running drift report: {window_id} ({len(current_df)} events)")
        try:
            result = run_evidently_report(self.reference_df, current_df, window_id)
            log_drift_result(result)

            drift_share = result["drift_share"]
            print(
                f"  Drift share: {drift_share:.2%} | "
                f"Dataset drift: {result['dataset_drift']}"
            )

            if result["dataset_drift"] or drift_share >= settings.drift_share_threshold:
                print(f"  DRIFT DETECTED (share={drift_share:.2%})")
                from src.monitoring.alert_handler import handle_drift_alert
                handle_drift_alert(result)
                return True

        except Exception as e:
            print(f"Drift report failed: {e}")

        return False
