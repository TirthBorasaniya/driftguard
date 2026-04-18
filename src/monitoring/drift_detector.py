"""Evidently AI drift detection: batch reports, feature drift analysis, and history tracking."""

import json
import sqlite3
from datetime import datetime

import pandas as pd
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DataDriftTable,
    DatasetDriftMetric,
)
from evidently.report import Report

from src.config import (
    BATCH_DIR,
    BATCH_FILES,
    CATEGORICAL_COLS,
    DB_PATH,
    DRIFT_DATASET_THRESHOLD,
    DRIFT_SHARE_THRESHOLD,
    FEATURE_COLS,
    NUMERIC_COLS,
    REFERENCE_FILE,
    REPORTS_DIR,
    TARGET_COL,
)


# ============= Column Mapping =============


def get_column_mapping():
    """
    Build Evidently ColumnMapping for the IEEE-CIS feature set.

    Returns
    -------
    mapping : ColumnMapping
        Column mapping with numeric and categorical feature lists.
    """
    return ColumnMapping(
        target=TARGET_COL,
        numerical_features=[c for c in NUMERIC_COLS if c in FEATURE_COLS],
        categorical_features=[c for c in CATEGORICAL_COLS if c in FEATURE_COLS],
    )


# ============= Drift Report Generation =============


def run_drift_report(reference_df, current_df, batch_name="batch"):
    """
    Generate an Evidently data drift report comparing current data against reference.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference dataset (from training period).
    current_df : pd.DataFrame
        Current production batch to check for drift.
    batch_name : str
        Identifier for this batch.

    Returns
    -------
    result : dict
        Drift summary with share, dataset_drift flag, per-feature results.
    report_path : str
        Path to saved HTML report.
    """
    column_mapping = get_column_mapping()

    # use only columns present in both dataframes
    common_cols = [c for c in FEATURE_COLS if c in reference_df.columns and c in current_df.columns]
    ref = reference_df[common_cols].copy()
    cur = current_df[common_cols].copy()

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
    ])

    report.run(reference_data=ref, current_data=cur, column_mapping=column_mapping)

    # save HTML report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"drift_report_{batch_name}_{timestamp}.html"
    report.save_html(str(report_path))

    # extract metrics from report dict
    report_dict = report.as_dict()
    result = _parse_drift_results(report_dict, batch_name)
    result["report_path"] = str(report_path)

    print(f"\nDrift Report: {batch_name}")
    print(f"  Dataset drift: {result['dataset_drift']}")
    print(f"  Drift share: {result['drift_share']:.2%}")
    print(f"  Drifted features: {result['n_drifted_features']}/{result['n_features']}")

    return result, str(report_path)


def _parse_drift_results(report_dict, batch_name):
    """
    Extract drift metrics from Evidently report dictionary.

    Parameters
    ----------
    report_dict : dict
        Output of report.as_dict().
    batch_name : str
        Batch identifier.

    Returns
    -------
    result : dict
        Parsed drift summary.
    """
    metrics = report_dict.get("metrics", [])

    dataset_drift = False
    drift_share = 0.0
    n_drifted = 0
    n_features = 0
    drifted_columns = []

    for metric in metrics:
        metric_id = metric.get("metric", "")
        metric_result = metric.get("result", {})

        if metric_id == "DatasetDriftMetric":
            dataset_drift = metric_result.get("dataset_drift", False)
            drift_share = metric_result.get("share_of_drifted_columns", 0.0)
            n_drifted = metric_result.get("number_of_drifted_columns", 0)
            n_features = metric_result.get("number_of_columns", 0)

        elif metric_id == "DataDriftTable":
            drift_by_columns = metric_result.get("drift_by_columns", {})
            for col_name, col_data in drift_by_columns.items():
                if col_data.get("drift_detected", False):
                    drifted_columns.append({
                        "column": col_name,
                        "drift_score": col_data.get("drift_score", 0),
                        "stattest_name": col_data.get("stattest_name", ""),
                    })

    return {
        "batch_name": batch_name,
        "dataset_drift": dataset_drift,
        "drift_share": drift_share,
        "n_drifted_features": n_drifted,
        "n_features": n_features,
        "drifted_columns": drifted_columns,
        "timestamp": datetime.now().isoformat(),
    }


# ============= Prediction Drift (Tier 1 Proxy) =============


def check_prediction_drift(reference_predictions, current_predictions):
    """
    Check for prediction distribution drift using KS test.

    This serves as a real-time proxy for concept drift when ground truth
    labels are delayed (common in fraud detection with chargeback lag).

    Parameters
    ----------
    reference_predictions : np.ndarray
        Prediction scores from reference period.
    current_predictions : np.ndarray
        Prediction scores from current period.

    Returns
    -------
    result : dict
        KS statistic and drift flag.
    """
    from scipy import stats

    ks_stat, p_value = stats.ks_2samp(reference_predictions, current_predictions)

    return {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "prediction_drift_detected": p_value < DRIFT_DATASET_THRESHOLD,
    }


# ============= Database Logging =============


def log_drift_report(result):
    """
    Persist drift report summary to the predictions database.

    Parameters
    ----------
    result : dict
        Drift report result from run_drift_report.
    """
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """INSERT INTO drift_reports
               (batch_name, drift_share, dataset_drift, n_drifted_features,
                n_features, report_path, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                result["batch_name"],
                result["drift_share"],
                int(result["dataset_drift"]),
                result["n_drifted_features"],
                result["n_features"],
                result.get("report_path", ""),
                result["timestamp"],
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_drift_history(limit=20):
    """
    Retrieve drift report history from database.

    Parameters
    ----------
    limit : int
        Maximum number of records.

    Returns
    -------
    history : list of dict
        Drift report records ordered by most recent first.
    """
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM drift_reports ORDER BY id DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


# ============= Main =============


def run_drift_detection():
    """Run drift detection on all production batches against reference data."""
    print("=" * 60)
    print("DriftGuard: Drift Detection")
    print("=" * 60)

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference data not found: {REFERENCE_FILE}. Run preprocessing first."
        )

    reference_df = pd.read_csv(REFERENCE_FILE)
    print(f"Reference data: {reference_df.shape}")

    results = []
    for batch_path in BATCH_FILES:
        if not batch_path.exists():
            print(f"  Skipping {batch_path.name}: not found")
            continue

        batch_name = batch_path.stem
        batch_df = pd.read_csv(batch_path)
        print(f"\nAnalyzing {batch_name}: {batch_df.shape}")

        result, report_path = run_drift_report(reference_df, batch_df, batch_name)
        log_drift_report(result)
        results.append(result)

    # also check any additional batches in the directory
    for batch_path in sorted(BATCH_DIR.glob("*.csv")):
        if batch_path in BATCH_FILES:
            continue
        batch_name = batch_path.stem
        batch_df = pd.read_csv(batch_path)
        print(f"\nAnalyzing {batch_name}: {batch_df.shape}")

        result, report_path = run_drift_report(reference_df, batch_df, batch_name)
        log_drift_report(result)
        results.append(result)

    print("\n" + "=" * 60)
    print(f"Drift detection complete. {len(results)} batches analyzed.")
    for r in results:
        status = "DRIFT DETECTED" if r["dataset_drift"] else "no drift"
        print(f"  {r['batch_name']}: {status} (share: {r['drift_share']:.2%})")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_drift_detection()
