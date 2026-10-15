"""E1 model quality: time-ordered, within-day, and random splits with ablations (SPEC 9)."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from driftguard.common.config import LoadedConfig, load_config
from driftguard.common.logging import get_logger
from driftguard.common.settings import REPO_ROOT
from driftguard.data.cicids2017 import load_dataset, model_matrix
from driftguard.data.registry import feature_names
from driftguard.training.evaluate import evaluate_scores
from driftguard.training.train import predict_scores, select_threshold, train_model

# ============= Constants =============

logger = get_logger("benchmark.e1")
DATASET = "cicids2017"


# ============= Splits =============


def blocked_validation_mask(n_rows: int, block_size: int, fraction: float) -> np.ndarray:
    """
    Mask of validation rows when whole contiguous blocks of the stream are held out.

    Every k-th block (k = round(1 / fraction)) is validation. Flows inside one attack campaign are
    near-duplicates of their temporal neighbors, so a random slice puts neighbors of every fit row
    into validation and makes it look perfect; holding out whole blocks removes that adjacency.

    Parameters
    ----------
    n_rows : int
        Number of training rows, in stream order.
    block_size : int
        Rows per block.
    fraction : float
        Approximate validation share.

    Returns
    -------
    mask : np.ndarray
        Boolean array of shape (n_rows,).
    """
    every = max(2, round(1.0 / fraction))
    return (np.arange(n_rows) // int(block_size)) % every == every - 1


def split_validation(
    frame: pd.DataFrame, method: str, fraction: float, seed: int, block_size: int = 5000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a training portion into fit and threshold-validation parts.

    Parameters
    ----------
    frame : pd.DataFrame
        Training portion; rows are put back in stream order (by index) before blocking.
    method : str
        `blocked` (whole contiguous blocks), `tail` (the newest `fraction` by event time), or
        `random_stratified` (kept for comparison; leaks temporal neighbors).
    fraction : float
        Validation share.
    seed : int
        Seed for the stratified split.
    block_size : int
        Rows per block for `blocked`.

    Returns
    -------
    fit, validation : tuple of pd.DataFrame
    """
    if method == "blocked":
        ordered = frame.sort_index()
        mask = blocked_validation_mask(len(ordered), block_size, fraction)
        return ordered.loc[~mask], ordered.loc[mask]
    if method == "tail":
        ordered = frame.sort_index()
        cut = int(len(ordered) * (1 - fraction))
        return ordered.iloc[:cut], ordered.iloc[cut:]
    if method == "random_stratified":
        fit, val = train_test_split(
            frame, test_size=fraction, stratify=frame["label"], random_state=seed
        )
        return fit, val
    raise ValueError(f"unknown validation method {method!r}")


def time_ordered_split(
    frame: pd.DataFrame, train_days: list[str], test_days: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on `train_days`, test on `test_days`."""
    return frame[frame["day"].isin(train_days)], frame[frame["day"].isin(test_days)]


def within_day_split(
    frame: pd.DataFrame, train_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within each time-sorted day, the first `train_fraction` trains and the rest tests."""
    train_parts, test_parts = [], []
    for _, day_frame in frame.groupby("day", sort=False):
        cut = int(len(day_frame) * train_fraction)
        train_parts.append(day_frame.iloc[:cut])
        test_parts.append(day_frame.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts)


def random_split(frame: pd.DataFrame, test_fraction: float, seed: int):
    """Stratified random split; an upper-bound reference only."""
    return train_test_split(
        frame, test_size=test_fraction, stratify=frame["label"], random_state=seed
    )


# ============= Protocol runner =============


def run_protocol(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_config: LoadedConfig,
    validation: dict[str, Any],
    target_recall: float,
) -> dict[str, Any]:
    """
    Fit on a training portion, choose the threshold on its validation slice, evaluate on test.

    Parameters
    ----------
    train : pd.DataFrame
        Training portion.
    test : pd.DataFrame
        Test portion.
    features : list[str]
        Model feature names.
    model_config : LoadedConfig
        Loaded `configs/model.yaml`.
    validation : dict[str, Any]
        Validation split settings (`method`, `fraction`, `block_size`).
    target_recall : float
        Recall target for the threshold.

    Returns
    -------
    result : dict
        Row counts, `scale_pos_weight`, `threshold`, validation metrics, and test metrics.
    """
    seed = int(model_config["seed"])
    fit, val = split_validation(
        train,
        validation["method"],
        float(validation["fraction"]),
        seed,
        int(validation.get("block_size", 5000)),
    )
    model, spw = train_model(
        model_matrix(fit, features), fit["label"].to_numpy(), model_config["lightgbm"], seed
    )
    val_scores = predict_scores(model, model_matrix(val, features))
    threshold = select_threshold(val["label"].to_numpy(), val_scores, target_recall)
    test_scores = predict_scores(model, model_matrix(test, features))
    # a deployable "recall 0.99" operating point: the threshold comes from the validation slice
    threshold_99 = select_threshold(val["label"].to_numpy(), val_scores, 0.99)
    at_validation_99 = evaluate_scores(test["label"].to_numpy(), test_scores, threshold_99)
    return {
        "n_features": len(features),
        "train_rows": len(fit),
        "train_positive": int(fit["label"].sum()),
        "validation_rows": len(val),
        "validation_positive": int(val["label"].sum()),
        "test_rows": len(test),
        "scale_pos_weight": spw,
        "threshold": threshold,
        "validation": evaluate_scores(val["label"].to_numpy(), val_scores, threshold),
        "test_at_validation_recall_0_99": {
            "threshold": threshold_99,
            "precision": at_validation_99["precision"],
            "recall": at_validation_99["recall"],
        },
        "test": evaluate_scores(
            test["label"].to_numpy(), test_scores, threshold, test["label_family"].to_numpy()
        ),
    }


# ============= Experiment =============


def run_e1(
    config: LoadedConfig,
    stream_feature_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """
    Run every E1 protocol and ablation.

    Parameters
    ----------
    config : LoadedConfig
        `configs/eval/e1.yaml`.
    stream_feature_fn : callable or None
        Returns the frame with stream-computed behavioral feature columns added; required for
        the stream-feature ablation.

    Returns
    -------
    metrics : dict
        Results keyed by protocol and ablation.
    """
    model_config = load_config(REPO_ROOT / config["model_config"])
    seed = int(model_config["seed"])
    target_recall = float(config["target_recall"])
    validation = config["validation"]
    protocols = config["protocols"]
    o_scratch = bool(config.get("o_run_from_scratch", False))
    features = feature_names(DATASET)

    frame, cleaning = load_dataset(o_run_from_scratch=o_scratch)
    metrics: dict[str, Any] = {
        "features": features,
        "cleaning_report": cleaning,
        "dataset_counts": summarize_counts(frame),
        "protocols": {},
    }
    to = protocols["time_ordered"]

    logger.info("e1_protocol", protocol="time_ordered")
    train, test = time_ordered_split(frame, to["train_days"], to["test_days"])
    metrics["protocols"]["time_ordered"] = run_protocol(
        train, test, features, model_config, validation, target_recall
    )
    logger.info("e1_protocol", protocol="within_day")
    train, test = within_day_split(frame, float(protocols["within_day"]["train_fraction"]))
    metrics["protocols"]["within_day"] = run_protocol(
        train, test, features, model_config, validation, target_recall
    )
    logger.info("e1_protocol", protocol="random_stratified")
    train, test = random_split(frame, float(protocols["random_stratified"]["test_fraction"]), seed)
    metrics["protocols"]["random_stratified_upper_bound"] = run_protocol(
        train, test, features, model_config, validation, target_recall
    )

    ablations: dict[str, Any] = {}
    if config["ablations"].get("o_drop_dst_port"):
        logger.info("e1_ablation", ablation="o_drop_dst_port")
        train, test = time_ordered_split(frame, to["train_days"], to["test_days"])
        reduced = [name for name in features if name != "dst_port"]
        ablations["o_drop_dst_port"] = run_protocol(
            train, test, reduced, model_config, validation, target_recall
        )

    policy_results = {}
    for policy in config["ablations"].get("attempted_policies", []):
        logger.info("e1_ablation", ablation="attempted_policy", policy=policy)
        policy_frame, policy_cleaning = load_dataset(
            attempted_policy=policy, o_run_from_scratch=o_scratch
        )
        train, test = time_ordered_split(policy_frame, to["train_days"], to["test_days"])
        result = run_protocol(train, test, features, model_config, validation, target_recall)
        result["cleaning_report"] = policy_cleaning
        policy_results[policy] = result
    if policy_results:
        ablations["attempted_policy"] = policy_results

    if config["ablations"].get("o_stream_features"):
        if stream_feature_fn is None:
            raise ValueError("o_stream_features requires a stream feature function (Redis)")
        logger.info("e1_ablation", ablation="stream_features")
        enriched = stream_feature_fn(frame)
        behavior = [col for col in enriched.columns if col not in frame.columns]
        train, test = time_ordered_split(enriched, to["train_days"], to["test_days"])
        ablations["stream_features"] = {
            "behavioral_features": behavior,
            "cicflowmeter_only": metrics["protocols"]["time_ordered"],
            "cicflowmeter_plus_behavioral": run_protocol(
                train, test, features + behavior, model_config, validation, target_recall
            ),
        }
    metrics["ablations"] = ablations
    metrics["v1_figure_trace"] = trace_v1_figure(metrics)
    return metrics


def trace_v1_figure(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Report, per protocol, the oracle precision at recall 0.99 and the precision and recall at the
    validation-chosen recall-0.99 threshold, for comparison with the unverified "98% precision at
    99% recall" figure (SPEC 9, E1). Only the validation-chosen numbers are deployable.
    """
    return {
        name: {
            "oracle_precision_at_recall_0_99": result["test"]["oracle_precision_at_recall_0_99"],
            "validated_precision": result["test_at_validation_recall_0_99"]["precision"],
            "validated_recall": result["test_at_validation_recall_0_99"]["recall"],
        }
        for name, result in metrics["protocols"].items()
    } | {"unverified_target_figure": {"precision": 0.98, "recall": 0.99}}


def summarize_counts(frame: pd.DataFrame) -> dict[str, Any]:
    """Return per-day row and attack counts plus per-family row counts."""
    grouped = frame.groupby("day", sort=False)["label"]
    return {
        "per_day": {day: {"rows": int(g.size), "attacks": int(g.sum())} for day, g in grouped},
        "per_family": {str(k): int(v) for k, v in frame["label_family"].value_counts().items()},
    }
