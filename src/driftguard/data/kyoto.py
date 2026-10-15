"""Kyoto 2006+ adapter over the AnoShift parquet subsets (SPEC 3.3)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from driftguard.common.config import LoadedConfig
from driftguard.common.logging import get_logger
from driftguard.common.settings import REPO_ROOT
from driftguard.data.ids import make_event_ids
from driftguard.data.registry import dataset_config

# ============= Constants =============

DATASET = "kyoto"
logger = get_logger("data.kyoto")


# ============= Column transforms =============


def parse_bins(values: pd.Series, column: str) -> np.ndarray:
    """
    Parse AnoShift discretized values `c<column><bin>` into integer bin indices.

    Raises
    ------
    ValueError
        If a value does not carry the expected column prefix.
    """
    prefix = f"c{column}"
    text = values.astype(str)
    if not text.str.startswith(prefix).all():
        bad = text[~text.str.startswith(prefix)].iloc[0]
        raise ValueError(f"column {column}: unexpected bin value {bad!r}")
    return text.str.slice(len(prefix)).astype(np.int64).to_numpy()


def encode_category(values: pd.Series, vocabulary: list[str]) -> tuple[np.ndarray, int]:
    """
    Ordinal codes from a fixed vocabulary; unseen values map to -1.

    Returns
    -------
    codes : np.ndarray
        int64 codes, shape (n,).
    n_unseen : int
        Rows whose value is outside the vocabulary.
    """
    lookup = {value: i for i, value in enumerate(vocabulary)}
    codes = values.astype(str).map(lookup).fillna(-1).astype(np.int64).to_numpy()
    return codes, int((codes == -1).sum())


def map_labels(values: pd.Series, labels_cfg: dict) -> pd.DataFrame:
    """
    Map AnoShift labels: normal -> 0; known and unknown attacks -> 1 with family `known`/`unknown`.

    Raises
    ------
    ValueError
        On a label outside the three documented values.
    """
    text = values.astype(str).str.strip()
    mapping = {labels_cfg["normal"]: (0, "normal"), labels_cfg["known_attack"]: (1, "known"),
               labels_cfg["unknown_attack"]: (1, "unknown")}
    unknown = set(text.unique()) - set(mapping)
    if unknown:
        raise ValueError(f"unexpected Kyoto labels: {sorted(unknown)}")
    return pd.DataFrame({"label": text.map(lambda v: mapping[v][0]).astype("int8"),
                         "label_family": text.map(lambda v: mapping[v][1])})


def check_no_leakage(columns: list[str], config: LoadedConfig) -> None:
    """Raise if any feature column is an excluded detector, IP, or port column."""
    excluded = {str(c) for c in config["excluded_columns"]}
    leaked = [c for c in columns if str(c) in excluded]
    if leaked:
        raise ValueError(f"leakage columns requested as features: {leaked}")


def transform(raw: pd.DataFrame, source_file: str, config: LoadedConfig) -> tuple[pd.DataFrame, dict]:
    """
    Convert one AnoShift frame into the adapter output.

    Returns
    -------
    frame : pd.DataFrame
        `event_id`, `dataset`, `source_file`, `row_index`, `event_time` (epoch ms), `month`,
        features (float32), `label`, `label_family`.
    report : dict
        Rows read, rows dropped for unparseable values, unseen categorical values per feature.
    """
    # parquet files after 2006 carry a non-contiguous pandas index; every column below is built
    # positionally, so a stale index would misalign labels with features
    raw = raw.reset_index(drop=True)
    raw.columns = [str(c) for c in raw.columns]
    specs = config["features"]
    check_no_leakage([spec["column"] for spec in specs], config)
    row_index = np.arange(len(raw), dtype=np.int64)
    features, unseen = {}, {}
    for spec in specs:
        values = raw[spec["column"]]
        if spec["kind"] == "bin":
            features[spec["name"]] = parse_bins(values, spec["column"]).astype(np.float64)
        elif spec["kind"] == "category":
            codes, n_unseen = encode_category(values, config["vocabularies"][spec["vocabulary"]])
            features[spec["name"]] = codes.astype(np.float64)
            unseen[spec["name"]] = n_unseen
        else:
            features[spec["name"]] = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    feature_frame = pd.DataFrame(features)
    times = pd.to_datetime(raw[config["timestamp_column"]], format=config["timestamp_format"],
                           utc=True, errors="coerce")
    labels = map_labels(raw[config["label_column"]], config["labels"])
    frame = pd.concat([
        pd.DataFrame({
            "dataset": DATASET,
            "source_file": source_file,
            "row_index": row_index,
            "event_time": (times.astype("int64") // 1_000_000).to_numpy(),
            "label": labels["label"],
            "label_family": labels["label_family"],
        }),
        feature_frame,
    ], axis=1)
    valid = times.notna().to_numpy() & np.isfinite(feature_frame.to_numpy()).all(axis=1)
    frame = frame.loc[valid].copy()
    names = [spec["name"] for spec in specs]
    frame[names] = frame[names].astype("float32")
    frame.insert(0, "event_id", make_event_ids(DATASET, source_file, frame["row_index"].to_numpy()))
    frame["month"] = pd.to_datetime(frame["event_time"], unit="ms", utc=True).dt.strftime("%Y-%m")
    report = {"rows_read": len(raw), "dropped_invalid": int((~valid).sum()),
              "rows_kept": len(frame), "unseen_categories": unseen}
    logger.info("kyoto_file_loaded", source_file=source_file, **report)
    return frame.reset_index(drop=True), report


# ============= Loading =============


def year_path(year: int, config: LoadedConfig) -> Path:
    """Path of one year's subset parquet."""
    return (REPO_ROOT / config["raw_dir"] / config["member_dir"]
            / config["file_pattern"].format(year=year))


def load_years(years: list[int], config: LoadedConfig | None = None,
               o_run_from_scratch: bool = False) -> pd.DataFrame:
    """
    Load and concatenate years in event-time order, with a parquet cache.

    Parameters
    ----------
    years : list of int
        Years to load.
    config : LoadedConfig or None
        Adapter config.
    o_run_from_scratch : bool
        Rebuild the cache.

    Returns
    -------
    frame : pd.DataFrame
        Adapter output sorted by event time, then source file and row index.
    """
    config = config or dataset_config(DATASET)
    cache_dir = REPO_ROOT / config["processed_dir"]
    stem = f"years_{'-'.join(str(y) for y in years)}"
    cache, report_path = cache_dir / f"{stem}.parquet", cache_dir / f"{stem}.report.json"
    if cache.exists() and not o_run_from_scratch:
        return pd.read_parquet(cache)
    frames, reports = [], {}
    for year in years:
        path = year_path(year, config)
        frame, reports[str(year)] = transform(pd.read_parquet(path), path.name, config)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["event_time", "source_file", "row_index"], kind="mergesort").reset_index(drop=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache, index=False)
    report_path.write_text(json.dumps(reports, indent=2))
    return combined


def load_stream(config: LoadedConfig | None = None, o_run_from_scratch: bool = False) -> pd.DataFrame:
    """The streamed years (2008 to 2015 by default) in event-time order."""
    config = config or dataset_config(DATASET)
    return load_years(list(config["stream_years"]), config, o_run_from_scratch)
