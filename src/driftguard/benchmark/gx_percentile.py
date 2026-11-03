"""Count real stream windows that leave the fitted percentile bounds (DECISIONS D-022)."""

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pandas as pd

from driftguard.common.config import load_config
from driftguard.common.settings import REPO_ROOT
from driftguard.data.registry import dataset_config, feature_names
from driftguard.monitor.windows import month_id
from driftguard.training.validate import GXValidator, load_bounds, physical_bounds

# ============= Constants =============

BETWEEN_EXPECTATION = "expect_column_values_to_be_between"
DEFAULT_CONFIG = "configs/eval/gx_percentile.yaml"


# ============= Windows =============


def count_windows(
    frame: pd.DataFrame, features: list[str], window_size: int
) -> Iterator[tuple[str, np.ndarray, int]]:
    """
    Yield consecutive full event-count windows of a stream-ordered frame.

    Parameters
    ----------
    frame : pd.DataFrame
        Frame in stream order with `event_time` (milliseconds) and the feature columns.
    features : list[str]
        Feature columns.
    window_size : int
        Events per window; a trailing partial window is not yielded, as in the monitor.

    Yields
    ------
    window_id : str
        `window_<index>`.
    values : np.ndarray
        Feature matrix, shape `(window_size, len(features))`.
    end_time_ms : int
        Latest event time in the window.
    """
    for index in range(len(frame) // window_size):
        chunk = frame.iloc[index * window_size : (index + 1) * window_size]
        yield (
            f"window_{index:05d}",
            chunk[features].to_numpy(dtype=np.float64),
            int(chunk["event_time"].max()),
        )


def month_windows(
    frame: pd.DataFrame, features: list[str]
) -> Iterator[tuple[str, np.ndarray, int]]:
    """Yield one window per UTC calendar month with the same fields as `count_windows`."""
    months = frame["event_time"].map(month_id)
    for month, chunk in frame.groupby(months, sort=True):
        yield (
            str(month),
            chunk[features].to_numpy(dtype=np.float64),
            int(chunk["event_time"].max()),
        )


# ============= Scan =============


def scan_windows(
    windows: Iterable[tuple[str, np.ndarray, int]],
    features: list[str],
    percentile: GXValidator,
    physical: GXValidator,
    reference_end_ms: int | None = None,
) -> dict[str, Any]:
    """
    Validate each window against the percentile bounds and the physical bounds.

    Parameters
    ----------
    windows : iterable
        `(window_id, values, end_time_ms)` tuples.
    features : list[str]
        Column names of `values`.
    percentile : GXValidator
        Suite on the fitted percentile bounds.
    physical : GXValidator
        Suite on the physical data-quality bounds.
    reference_end_ms : int or None
        Event time at which the reference (training) period ends; windows ending at or before it
        are counted separately as in-reference windows.

    Returns
    -------
    summary : dict
        `windows` (one entry per window) and totals, including windows failing only the
        percentile bounds, which are the windows D-022 stops from blocking retraining.
    """
    rows = []
    for window_id, values, end_ms in windows:
        frame = pd.DataFrame(values, columns=features)
        pct = percentile.validate(frame)
        phys = physical.validate(frame)
        rows.append(
            {
                "window_id": window_id,
                "rows": int(len(frame)),
                "end_event_time_ms": end_ms,
                "in_reference_period": reference_end_ms is not None and end_ms <= reference_end_ms,
                "percentile_failed": not pct.success,
                "percentile_features_outside": sum(
                    1 for name in pct.failed if name.startswith(BETWEEN_EXPECTATION)
                ),
                "physical_failed": not phys.success,
            }
        )
    outside = [r for r in rows if r["percentile_failed"] and not r["in_reference_period"]]
    return {
        "windows": rows,
        "n_windows": len(rows),
        "n_reference_windows": sum(r["in_reference_period"] for r in rows),
        "n_percentile_failed": sum(r["percentile_failed"] for r in rows),
        "n_percentile_failed_after_reference": len(outside),
        "n_physical_failed": sum(r["physical_failed"] for r in rows),
        "n_failed_percentile_only": sum(
            r["percentile_failed"] and not r["physical_failed"] for r in rows
        ),
        "first_percentile_failure_after_reference": outside[0]["window_id"] if outside else None,
    }


# ============= Experiment =============


def run_gx_percentile(config_path: str = DEFAULT_CONFIG) -> dict[str, Any]:
    """
    Run the scan for every configured dataset on the real streams.

    Parameters
    ----------
    config_path : str
        Eval config listing the datasets.

    Returns
    -------
    metrics : dict
        Per-dataset scan summaries and the suite settings used.
    """
    config = load_config(REPO_ROOT / config_path)
    policy = load_config(REPO_ROOT / config["policy_config"])
    metrics: dict[str, Any] = {"window_size": int(policy["window_size"]), "datasets": {}}
    for dataset in config["datasets"]:
        gx = dataset_config(dataset)["gx"]
        features = feature_names(dataset)
        bounds_path = REPO_ROOT / "configs" / "data" / f"{dataset}_gx_bounds.json"
        percentile = GXValidator(
            load_bounds(bounds_path),
            float(gx["mostly"]),
            int(gx["min_window_rows"]),
            name=f"{dataset}_pct",
        )
        physical = GXValidator(
            physical_bounds(dataset),
            float(gx["monitor_mostly"]),
            int(gx["min_window_rows"]),
            name=f"{dataset}_phys",
        )
        if dataset == "cicids2017":
            from driftguard.data.cicids2017 import load_dataset

            stream, _ = load_dataset()
            reference, _ = load_dataset(days=list(gx["reference_days"]))
            reference_end = int(reference["event_time"].max())
            windows = count_windows(stream, features, int(policy["window_size"]))
        else:
            from driftguard.data.kyoto import load_stream

            reference_end = None
            windows = month_windows(load_stream(), features)
        summary = scan_windows(windows, features, percentile, physical, reference_end)
        summary["bounds_file"] = str(bounds_path.relative_to(REPO_ROOT))
        summary["percentile_mostly"] = float(gx["mostly"])
        metrics["datasets"][dataset] = summary
    return metrics
