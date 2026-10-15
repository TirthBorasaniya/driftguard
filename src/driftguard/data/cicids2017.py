"""CICIDS2017 adapter: load, clean, label, and order flows for replay and training (SPEC 3.2)."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.common.config import LoadedConfig
from driftguard.common.logging import get_logger
from driftguard.common.settings import REPO_ROOT
from driftguard.data.ids import make_event_ids
from driftguard.data.registry import dataset_config, feature_specs

# ============= Constants =============

DATASET = "cicids2017"
ATTEMPTED_POLICIES = ("benign", "attack", "drop")
META_COLUMNS = [
    "event_id",
    "dataset",
    "source_file",
    "row_index",
    "event_time",
    "day",
    "src_ip",
    "dst_ip",
    "label",
    "label_family",
    "attempted",
]
NON_ASCII_DASH = re.compile(r"\s*[–—�\x96]\s*")
logger = get_logger("data.cicids2017")


# ============= Label mapping =============


def normalize_label(raw: str) -> str:
    """Strip whitespace and replace the non-ASCII dashes found in the original release."""
    return NON_ASCII_DASH.sub(" - ", str(raw).strip())


def map_labels(
    raw_labels: pd.Series,
    attempted_policy: str,
    benign_label: str = "BENIGN",
    attempted_suffix: str = " - Attempted",
) -> pd.DataFrame:
    """
    Map raw CICIDS2017 labels to binary labels and families.

    Parameters
    ----------
    raw_labels : pd.Series
        Raw label strings, for example `BENIGN`, `DoS Hulk`, `FTP-Patator - Attempted`.
    attempted_policy : str
        `benign` maps attempted rows to 0, `attack` maps them to 1, `drop` marks them for removal.
    benign_label : str
        Label string of benign traffic.
    attempted_suffix : str
        Suffix marking attempted attacks.

    Returns
    -------
    mapped : pd.DataFrame
        Columns `label` (int8), `label_family` (str, the attack name without the suffix, or
        `BENIGN`), `attempted` (bool), and `keep` (bool, False only under policy `drop`).
    """
    if attempted_policy not in ATTEMPTED_POLICIES:
        raise ValueError(f"attempted_policy must be one of {ATTEMPTED_POLICIES}")
    labels = raw_labels.astype(str).map(normalize_label)
    attempted = labels.str.endswith(attempted_suffix)
    family = labels.where(~attempted, labels.str.slice(0, -len(attempted_suffix)))
    is_benign = labels == benign_label
    label = (~is_benign).astype("int8")
    keep = pd.Series(True, index=labels.index)
    if attempted_policy == "benign":
        label = label.where(~attempted, 0).astype("int8")
    elif attempted_policy == "drop":
        keep = ~attempted
    return pd.DataFrame(
        {"label": label, "label_family": family, "attempted": attempted, "keep": keep}
    )


# ============= Loading =============


def _source_config(config: LoadedConfig, source: str) -> dict:
    """Return the per-source block of the adapter config."""
    if source not in config["sources"]:
        raise ValueError(f"unknown CICIDS2017 source {source!r}")
    return config["sources"][source]


def parse_timestamps(values: pd.Series, timestamp_format: str | None) -> np.ndarray:
    """
    Parse flow start timestamps into epoch milliseconds.

    The capture timestamps carry no zone; they are interpreted as UTC so event time is stable
    across hosts. The original release uses day-first dates with minute resolution.

    Returns
    -------
    event_time : np.ndarray
        int64 milliseconds since the epoch, shape (n_rows,).
    """
    if timestamp_format:
        parsed = pd.to_datetime(values, format=timestamp_format, utc=True)
    else:
        parsed = pd.to_datetime(values, dayfirst=True, utc=True, format="mixed")
    return (parsed.astype("int64") // 1_000_000).to_numpy()


def load_day(
    day: str,
    config: LoadedConfig | None = None,
    source: str | None = None,
    attempted_policy: str | None = None,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load, label, and clean one day of CICIDS2017.

    Parameters
    ----------
    day : str
        Day name from `day_order`.
    config : LoadedConfig or None
        Adapter config; defaults to `configs/data/cicids2017.yaml`.
    source : str or None
        `improved` or `original`; defaults to the configured source.
    attempted_policy : str or None
        Overrides the configured `attempted_policy`.
    nrows : int or None
        Read only the first rows (tests and smoke runs).

    Returns
    -------
    frame : pd.DataFrame
        One row per kept flow: META_COLUMNS plus the feature columns (float32).
    report : dict
        Row accounting: `rows_read`, `dropped_attempted`, `dropped_nonfinite`, `rows_kept`.
    """
    config = config or dataset_config(DATASET)
    source = source or config["source"]
    policy = attempted_policy or config["attempted_policy"]
    src = _source_config(config, source)
    file_name = src["files"][day]
    path = REPO_ROOT / src["raw_dir"] / file_name
    specs = feature_specs(DATASET, source)
    cols = src["columns"]

    raw = pd.read_csv(path, encoding=src["encoding"], low_memory=False, nrows=nrows)
    raw.columns = [str(c).strip() for c in raw.columns]
    # the original release repeats "Fwd Header Length"; pandas suffixes the duplicate with ".1"
    row_index = np.arange(len(raw), dtype=np.int64)

    features = pd.DataFrame(
        {
            spec["name"]: pd.to_numeric(raw[spec[source]], errors="coerce").astype("float64")
            for spec in specs
        }
    )
    labels = map_labels(raw[cols["label"]], policy, config["benign_label"], config["attempted_suffix"])
    frame = pd.DataFrame(
        {
            "dataset": DATASET,
            "source_file": file_name,
            "row_index": row_index,
            "event_time": parse_timestamps(raw[cols["timestamp"]], src["timestamp_format"]),
            "day": day,
            "src_ip": raw[cols["src_ip"]].astype(str).str.strip(),
            "dst_ip": raw[cols["dst_ip"]].astype(str).str.strip(),
            "label": labels["label"],
            "label_family": labels["label_family"],
            "attempted": labels["attempted"],
        }
    )
    frame = pd.concat([frame, features], axis=1)

    rows_read = len(frame)
    keep = labels["keep"].to_numpy()
    dropped_attempted = int((~keep).sum())
    frame = frame.loc[keep]
    frame, dropped_nonfinite = drop_nonfinite(frame, [spec["name"] for spec in specs])
    frame[[spec["name"] for spec in specs]] = frame[[spec["name"] for spec in specs]].astype(
        "float32"
    )
    frame.insert(0, "event_id", make_event_ids(DATASET, file_name, frame["row_index"].to_numpy()))
    report = {
        "rows_read": rows_read,
        "dropped_attempted": dropped_attempted,
        "dropped_nonfinite": dropped_nonfinite,
        "rows_kept": len(frame),
    }
    logger.info("cicids_day_loaded", day=day, source=source, **report)
    return frame.reset_index(drop=True), report


def drop_nonfinite(frame: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, int]:
    """
    Replace infinities with NaN and drop rows with NaN in any allowlisted feature.

    Returns
    -------
    cleaned : pd.DataFrame
        Rows with all-finite features.
    n_dropped : int
        Number of rows removed; callers record it, rows are never dropped silently.
    """
    values = frame[feature_cols].to_numpy(dtype=np.float64)
    finite_rows = np.isfinite(values).all(axis=1)
    return frame.loc[finite_rows], int((~finite_rows).sum())


def order_stream(frame: pd.DataFrame, day_order: list[str]) -> pd.DataFrame:
    """Sort flows by day order, then flow start time, then source row index."""
    day_rank = frame["day"].map({day: i for i, day in enumerate(day_order)})
    ordered = frame.assign(_day_rank=day_rank).sort_values(
        ["_day_rank", "event_time", "row_index"], kind="mergesort"
    )
    return ordered.drop(columns="_day_rank").reset_index(drop=True)


def load_dataset(
    config: LoadedConfig | None = None,
    source: str | None = None,
    attempted_policy: str | None = None,
    days: list[str] | None = None,
    o_run_from_scratch: bool = False,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load every configured day in stream order, with a parquet cache.

    Parameters
    ----------
    config : LoadedConfig or None
        Adapter config.
    source : str or None
        `improved` or `original`.
    attempted_policy : str or None
        Overrides the configured policy.
    days : list of str or None
        Subset of days; defaults to `day_order`.
    o_run_from_scratch : bool
        Ignore and rebuild the parquet cache.
    nrows : int or None
        Per-day row limit; disables the cache.

    Returns
    -------
    frame : pd.DataFrame
        All flows in stream order.
    cleaning_report : dict
        Per-day row accounting from `load_day`, keyed by day.
    """
    config = config or dataset_config(DATASET)
    source = source or config["source"]
    policy = attempted_policy or config["attempted_policy"]
    days = days or list(config["day_order"])
    cache_dir = REPO_ROOT / config["processed_dir"]
    stem = f"{source}_{policy}_{'-'.join(days)}"
    cache_path, report_path = cache_dir / f"{stem}.parquet", cache_dir / f"{stem}.report.json"

    if nrows is None and not o_run_from_scratch and cache_path.exists() and report_path.exists():
        return pd.read_parquet(cache_path), json.loads(report_path.read_text())

    frames, report = [], {}
    for day in days:
        frame, report[day] = load_day(day, config, source, policy, nrows=nrows)
        frames.append(frame)
    combined = order_stream(pd.concat(frames, ignore_index=True), list(config["day_order"]))
    if nrows is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(cache_path, index=False)
        report_path.write_text(json.dumps(report, indent=2))
    return combined, report


def model_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Return the float32 model input matrix for the given feature columns, shape (n, f)."""
    excluded = set(META_COLUMNS) | set(dataset_config(DATASET)["excluded_columns"])
    leaked = [col for col in feature_cols if col in excluded]
    if leaked:
        raise ValueError(f"excluded columns requested as features: {leaked}")
    return frame[feature_cols].to_numpy(dtype=np.float32)


def write_cleaning_report(report: dict, path: Path) -> None:
    """Persist per-day row accounting next to a results file."""
    path.write_text(json.dumps(report, indent=2))
