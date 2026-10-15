"""E2 drift detection benchmark: delay, false alarms, and misses per detector (SPEC 9, E2)."""

import zlib
from dataclasses import dataclass
from typing import Any

import fakeredis
import numpy as np
import pandas as pd

from driftguard.benchmark.injector import inject
from driftguard.common.config import load_config
from driftguard.common.logging import get_logger
from driftguard.common.settings import REPO_ROOT
from driftguard.data.registry import dataset_config, feature_names
from driftguard.monitor.main import MonitorCore
from driftguard.monitor.policy import CHANGE_SIGNALS, THRESHOLD_SIGNALS
from driftguard.monitor.reference import Reference, build_reference
from driftguard.training.train import predict_scores, select_threshold, train_model
from driftguard.training.validate import GXValidator, physical_bounds

# ============= Constants =============

logger = get_logger("benchmark.e2")
DETECTORS = (*THRESHOLD_SIGNALS, *CHANGE_SIGNALS)
POLICY = "policy"


# ============= Base data =============


@dataclass
class BaseData:
    """Stationary pool, reference model, and reference profile for one dataset."""

    dataset: str
    features: list[str]
    pool: pd.DataFrame
    attack_pool: pd.DataFrame
    reference_rows: pd.DataFrame
    reference: Reference
    model: Any
    threshold: float
    gx: Any = None


def load_base_frame(dataset: str, base_cfg: dict) -> pd.DataFrame:
    """Load the rows that make up the stationary base for a dataset."""
    if dataset == "cicids2017":
        from driftguard.data.cicids2017 import load_dataset

        frame, _ = load_dataset(days=list(base_cfg["days"]))
        return frame
    from driftguard.data.kyoto import load_years

    return load_years(list(base_cfg["years"]))


def prepare_base(
    dataset: str,
    frame: pd.DataFrame,
    config: dict,
    model_config: dict,
    policy_config: dict,
    seed: int = 0,
) -> BaseData:
    """
    Shuffle the base rows (making them stationary by construction), train the reference model on
    `train_fraction` of them, and keep the rest as the pool segments are drawn from.

    Parameters
    ----------
    dataset : str
        Dataset name.
    frame : pd.DataFrame
        All rows of the base period.
    config : dict
        `configs/eval/e2.yaml` content (uses `train_fraction`).
    model_config : dict
        `configs/model.yaml` content.
    policy_config : dict
        `configs/drift_policy.yaml` content.
    seed : int
        Seed for the shuffle.

    Returns
    -------
    base : BaseData
        Pool, attack pool, reference rows, drift reference, trained model, threshold, and GX
        validator.
    """
    features = feature_names(dataset)
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(len(shuffled) * float(config["train_fraction"]))
    train, pool = shuffled.iloc[:n_train], shuffled.iloc[n_train:].reset_index(drop=True)
    n_val = max(1, len(train) // 10)
    fit, val = train.iloc[n_val:], train.iloc[:n_val]
    model, _ = train_model(
        fit[features].to_numpy(np.float32),
        fit["label"].to_numpy(),
        model_config["lightgbm"],
        int(model_config["seed"]),
    )
    val_scores = predict_scores(model, val[features].to_numpy(np.float32))
    threshold = select_threshold(
        val["label"].to_numpy(), val_scores, float(model_config["target_recall"])
    )
    reference_rows = train[features].copy()
    reference_rows["score"] = predict_scores(model, reference_rows[features].to_numpy(np.float32))
    reference = build_reference(
        "benchmark",
        reference_rows,
        features,
        int(policy_config["psi_bins"]),
        int(policy_config["evidently_max_rows"]),
    )
    # the deployed monitor blocks windows that fail its quality suite, so E2's policy must too
    gx_cfg = dataset_config(dataset)["gx"]
    gx = GXValidator(
        physical_bounds(dataset),
        float(gx_cfg["monitor_mostly"]),
        int(gx_cfg["min_window_rows"]),
        name=f"{dataset}_e2",
    )
    return BaseData(
        dataset,
        features,
        pool,
        pool[pool["label"] == 1],
        reference_rows,
        reference,
        model,
        threshold,
        gx,
    )


# ============= One run =============


def label_arrivals(
    n: int, coverage: float, delay_min: int, delay_max: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Event index at which each row's label arrives; -1 for rows that never get a label.

    Returns
    -------
    arrival : np.ndarray
        int64, shape (n,).
    """
    covered = rng.random(n) < coverage
    delay = rng.integers(delay_min, delay_max + 1, size=n)
    return np.where(covered, np.arange(n) + delay, -1).astype(np.int64)


def run_segment(
    base: BaseData,
    segment: pd.DataFrame,
    policy_config: dict,
    coverage: float,
    delay: dict,
    chunk_size: int,
    rng: np.random.Generator,
) -> MonitorCore:
    """
    Stream a (possibly shifted) segment through a fresh MonitorCore and return it.

    Parameters
    ----------
    base : BaseData
        Prepared base data.
    segment : pd.DataFrame
        Rows to stream, in order (possibly with an injected shift).
    policy_config : dict
        `configs/drift_policy.yaml` content with the window size set.
    coverage : float
        Fraction of rows that eventually receive a label.
    delay : dict
        Label delay bounds, `{min, max}` in events.
    chunk_size : int
        Events fed to the monitor per step.
    rng : np.random.Generator
        Random generator for label arrival.

    Returns
    -------
    core : MonitorCore
        The monitor after the segment, holding signal and decision logs.
    """
    core = MonitorCore(
        base.dataset,
        policy_config,
        fakeredis.FakeRedis(decode_responses=True),
        gx_validator=base.gx,
        window_mode="count",
    )
    core.set_reference(base.reference, o_promotion=False)
    matrix = segment[base.features].to_numpy(np.float64)
    scores = predict_scores(base.model, matrix.astype(np.float32))
    decisions = (scores >= base.threshold).astype(int)
    labels = segment["label"].to_numpy().astype(int)
    event_ids = segment["event_id"].astype(str).to_numpy()
    arrival = label_arrivals(len(segment), coverage, int(delay["min"]), int(delay["max"]), rng)
    order = np.argsort(np.where(arrival < 0, np.iinfo(np.int64).max, arrival), kind="stable")
    pointer = 0
    for start in range(0, len(segment), chunk_size):
        end = min(start + chunk_size, len(segment))
        times = list(range(start, end))
        core.on_flow_vectors(times, matrix[start:end].tolist())
        core.on_predictions(
            [
                {
                    "event_id": event_ids[i],
                    "event_time": i,
                    "score": float(scores[i]),
                    "decision": int(decisions[i]),
                }
                for i in range(start, end)
            ]
        )
        due = []
        while pointer < len(order) and 0 <= arrival[order[pointer]] < end:
            i = order[pointer]
            due.append({"event_id": event_ids[i], "label": int(labels[i])})
            pointer += 1
        core.on_labels(due)
    return core


def window_flags(core: MonitorCore, thresholds: dict[str, float]) -> dict[str, list[bool]]:
    """Per-detector flag sequence over windows, plus the policy's emit sequence."""
    flags: dict[str, list[bool]] = {name: [] for name in DETECTORS}
    for _, signals in core.signal_log:
        for name in THRESHOLD_SIGNALS:
            flags[name].append(signals[name] > thresholds[name])
        for name in CHANGE_SIGNALS:
            flags[name].append(signals.get(name, 0.0) >= 1.0)
    flags[POLICY] = [decision.emit for decision in core.decisions]
    return flags


def score_flags(flags: list[bool], onset_window: int, horizon: int) -> dict[str, Any]:
    """
    Detection delay (windows), pre-onset false alarms, and miss for one flag sequence.

    A detection is the first flagged window at or after the onset window; it is a miss when that
    window is more than `horizon` windows after onset (or absent).
    """
    pre = flags[:onset_window]
    delay = next((i - onset_window for i in range(onset_window, len(flags)) if flags[i]), None)
    missed = delay is None or delay > horizon
    return {
        "delay": None if missed else delay,
        "missed": missed,
        "false_alarms": int(sum(pre)),
        "pre_windows": len(pre),
    }


# ============= Aggregation =============


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Median and IQR delay over detected runs, miss rate, false alarms per 100 windows."""
    delays = [r["delay"] for r in runs if r["delay"] is not None]
    pre = sum(r["pre_windows"] for r in runs)
    return {
        "runs": len(runs),
        "detected": len(delays),
        "median_delay_windows": float(np.median(delays)) if delays else None,
        "iqr_delay_windows": (
            [float(np.percentile(delays, 25)), float(np.percentile(delays, 75))] if delays else None
        ),
        "miss_rate": sum(r["missed"] for r in runs) / len(runs) if runs else None,
        "false_alarms_per_100_windows": 100.0 * sum(r["false_alarms"] for r in runs) / pre
        if pre
        else None,
    }


def summarize_gx(runs: list[dict[str, int]]) -> dict[str, float | None]:
    """
    GX gate statistics for one shift: failures on clean windows and the share blocked after onset.

    A window that fails the suite never emits a drift event, so a shift that pushes more than 1%
    of a feature outside its fitted bounds is blocked as a data-quality failure; the combined
    policy rows include that block.
    """
    before = sum(r["windows_before_onset"] for r in runs)
    after = sum(r["windows_after_onset"] for r in runs)
    return {
        "failures_per_100_windows_before_onset": (
            100.0 * sum(r["failures_before_onset"] for r in runs) / before if before else None
        ),
        "share_of_windows_blocked_after_onset": (
            sum(r["blocked_after_onset"] for r in runs) / after if after else None
        ),
    }


def shift_grid(config: dict, dataset: str) -> list[tuple[str, str, dict]]:
    """
    Every (shift type, magnitude label, params) combination for a dataset.

    Parameters
    ----------
    config : dict
        `configs/eval/e2.yaml` content.
    dataset : str
        Dataset name.

    Returns
    -------
    grid : list of tuple
        (shift type, magnitude label, parameters) for every shift and magnitude.
    """
    shifts = config["shifts"]
    grid = []
    cov = shifts["covariate"]
    for factor in cov["factors"]:
        grid.append(
            (
                "covariate",
                f"factor={factor}",
                {
                    "features": cov["features"][dataset],
                    "factor": factor,
                    "noise_std": cov["noise_std"],
                },
            )
        )
    for rate in shifts["prior"]["attack_rate_after"]:
        grid.append(("prior", f"attack_rate_after={rate}", {"attack_rate_after": rate}))
    concept = shifts["concept"]
    grid.append(
        (
            "concept",
            f"flip_fraction={concept['flip_fraction']}",
            {
                "region_rule": concept["region_rule"][dataset],
                "flip_fraction": concept["flip_fraction"],
            },
        )
    )
    return grid


def run_dataset(base: BaseData, config: dict, policy_config: dict) -> dict[str, Any]:
    """
    Run every shift, magnitude, coverage, and seed for one dataset.

    Parameters
    ----------
    base : BaseData
        Prepared base data.
    config : dict
        `configs/eval/e2.yaml` content.
    policy_config : dict
        `configs/drift_policy.yaml` content.

    Returns
    -------
    results : dict
        Per label coverage and shift: detector summaries, GX statistics, and onset manifests.
    """
    size, n_windows = int(config["window_size"]), int(config["n_windows"])
    # the benchmark's window size governs the monitor core's count windows
    policy_config = {**policy_config, "window_size": size}
    low, high = config["onset_window_range"]
    horizon = int(config["miss_horizon_windows"])
    n_rows = size * n_windows
    if len(base.pool) < n_rows:
        raise ValueError(f"base pool has {len(base.pool)} rows; {n_rows} needed per segment")
    results: dict[str, Any] = {}
    for coverage in config["label_coverages"]:
        cov_key = f"label_coverage={coverage}"
        results[cov_key] = {}
        for shift_type, magnitude, params in shift_grid(config, base.dataset):
            per_detector: dict[str, list[dict]] = {name: [] for name in (*DETECTORS, POLICY)}
            gx_runs: list[dict[str, int]] = []
            manifests = []
            for seed in range(int(config["n_seeds"])):
                rng = np.random.default_rng(
                    [seed, int(coverage * 100), zlib.crc32(f"{shift_type}:{magnitude}".encode())]
                )
                segment = base.pool.sample(n_rows, random_state=int(rng.integers(1 << 31)))
                segment = segment.reset_index(drop=True)
                onset_window = int(rng.integers(low, high + 1))
                onset = onset_window * size + int(rng.integers(0, size))
                shifted, manifest = inject(
                    segment,
                    shift_type,
                    onset,
                    params,
                    int(rng.integers(1 << 31)),
                    base.reference_rows,
                    base.attack_pool,
                )
                core = run_segment(
                    base,
                    shifted,
                    policy_config,
                    float(coverage),
                    config["label_delay"],
                    int(config["chunk_size"]),
                    rng,
                )
                flags = window_flags(core, policy_config["thresholds"])
                for name, sequence in flags.items():
                    per_detector[name].append(score_flags(sequence, onset // size, horizon))
                gx_failed = [d.data_quality_failed for d in core.decisions]
                onset_window = onset // size
                before, after = gx_failed[:onset_window], gx_failed[onset_window:]
                gx_runs.append(
                    {
                        "failures_before_onset": sum(before),
                        "windows_before_onset": len(before),
                        "blocked_after_onset": sum(after),
                        "windows_after_onset": len(after),
                    }
                )
                manifests.append(
                    {
                        "seed": seed,
                        "onset": manifest.onset,
                        "onset_window": onset // size,
                        "rows_modified": manifest.rows_modified,
                    }
                )
                logger.info(
                    "e2_run",
                    dataset=base.dataset,
                    shift=shift_type,
                    magnitude=magnitude,
                    coverage=coverage,
                    seed=seed,
                )
            results[cov_key][f"{shift_type}:{magnitude}"] = {
                "detectors": {name: summarize(runs) for name, runs in per_detector.items()},
                "gx": summarize_gx(gx_runs),
                "manifests": manifests,
            }
    return results


def run_e2(config_path: str) -> dict[str, Any]:
    """Run E2 for every configured dataset."""
    config = load_config(config_path).data
    policy_config = load_config(REPO_ROOT / config["policy_config"]).data
    model_config = load_config(REPO_ROOT / config["model_config"]).data
    metrics: dict[str, Any] = {
        "policy_version": policy_config["version"],
        "window_size": config["window_size"],
        "datasets": {},
    }
    for dataset in config["datasets"]:
        frame = load_base_frame(dataset, config["base"][dataset])
        base = prepare_base(dataset, frame, config, model_config, policy_config)
        metrics["datasets"][dataset] = {
            "base_rows": len(frame),
            "base_attack_rate": float(frame["label"].mean()),
            "results": run_dataset(base, config, policy_config),
        }
    return metrics
