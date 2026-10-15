"""Retraining logic shared by the Prefect flow and the bootstrap script (SPEC 6.2)."""

import io
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driftguard.common.logging import get_logger
from driftguard.common.mlflow_client import (
    CHALLENGER,
    CHAMPION,
    FEATURES_FILE,
    MODEL_FILE,
    REFERENCE_FILE,
    features_json,
    model_name,
)
from driftguard.training.evaluate import evaluate_scores
from driftguard.training.export import onnx_scores, onnx_session, parity_check, to_onnx_bytes
from driftguard.training.train import (
    MIN_THRESHOLD_HEADROOM,
    predict_scores,
    select_threshold,
    threshold_headroom,
    train_model,
)
from driftguard.training.window import WindowSplits, data_hash

# ============= Constants =============

logger = get_logger("training.flow")
STATUS_REJECTED_DQ = "rejected_data_quality"
STATUS_REGISTERED = "registered"
STATUS_INSUFFICIENT = "rejected_insufficient_data"


# ============= Packaging =============


@dataclass
class TrainedModel:
    """A trained, exported, parity-checked model and its artifacts."""

    features: list[str]
    onnx_bytes: bytes
    threshold: float
    scale_pos_weight: float
    parity_max_abs_diff: float
    validation_metrics: dict[str, Any]
    reference: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)

    def artifacts(self) -> dict[str, bytes]:
        """Files stored under the registered version's `model/` directory."""
        buffer = io.BytesIO()
        self.reference.to_parquet(buffer, index=False)
        return {
            MODEL_FILE: self.onnx_bytes,
            FEATURES_FILE: features_json(self.features),
            REFERENCE_FILE: buffer.getvalue(),
        }


def train_and_package(
    splits: WindowSplits,
    features: list[str],
    model_config: dict[str, Any],
    reference_max_rows: int,
) -> TrainedModel:
    """
    Train on the fit slice, choose the threshold on the validation slice, export to ONNX, check
    parity, and build the drift reference (training-window features plus model scores).

    Parameters
    ----------
    splits : WindowSplits
        Time-ordered fit, validation, and holdout slices.
    features : list of str
        Model input columns.
    model_config : dict
        `configs/model.yaml` content.
    reference_max_rows : int
        Size cap of the stored reference sample.

    Returns
    -------
    trained : TrainedModel
    """
    seed = int(model_config["seed"])
    x_fit = splits.fit[features].to_numpy(dtype=np.float32)
    model, spw = train_model(x_fit, splits.fit["label"].to_numpy(), model_config["lightgbm"], seed)
    val_scores = predict_scores(model, splits.validation[features].to_numpy(dtype=np.float32))
    threshold = select_threshold(
        splits.validation["label"].to_numpy(), val_scores, float(model_config["target_recall"])
    )
    headroom = threshold_headroom(threshold)
    if headroom < MIN_THRESHOLD_HEADROOM:
        logger.warning("threshold_saturated", threshold=threshold, headroom=headroom)
    onnx_cfg = model_config["onnx"]
    payload = to_onnx_bytes(
        model, len(features), int(onnx_cfg["split"]), int(onnx_cfg["target_opset"])
    )
    observed = parity_check(
        model,
        onnx_session(payload),
        x_fit,
        int(onnx_cfg["parity_rows"]),
        float(onnx_cfg["parity_max_abs_diff"]),
        seed,
    )
    training = pd.concat([splits.fit, splits.validation], ignore_index=True)
    if len(training) > reference_max_rows:
        training = training.sample(reference_max_rows, random_state=seed)
    reference = training[features].copy()
    reference["score"] = predict_scores(model, reference[features].to_numpy(dtype=np.float32))
    return TrainedModel(
        features=features,
        onnx_bytes=payload,
        threshold=threshold,
        scale_pos_weight=spw,
        parity_max_abs_diff=observed,
        validation_metrics={
            **evaluate_scores(splits.validation["label"].to_numpy(), val_scores, threshold),
            "threshold_headroom": headroom,
        },
        reference=reference.reset_index(drop=True),
        params={**model_config["lightgbm"], "seed": seed, "scale_pos_weight": spw},
    )


# ============= Evaluation =============


def score_with_onnx(model_dir: Path, frame: pd.DataFrame) -> np.ndarray:
    """Score a frame with a downloaded model directory, selecting its features by name."""
    features = json.loads((model_dir / FEATURES_FILE).read_text())
    session = onnx_session(model_dir / MODEL_FILE)
    return onnx_scores(session, frame[features].to_numpy(dtype=np.float32))


def compare_on_holdout(
    holdout: pd.DataFrame,
    challenger: TrainedModel,
    champion_dir: Path | None,
    champion_threshold: float | None,
) -> dict[str, Any]:
    """
    Score champion and challenger on the same holdout (the newest data) - SPEC 6.2 step 5.

    Returns
    -------
    comparison : dict
        `challenger` and `champion` metric dicts (champion None when absent).
    """
    y = holdout["label"].to_numpy()
    families = holdout["label_family"].to_numpy()
    challenger_scores = onnx_scores(
        onnx_session(challenger.onnx_bytes), holdout[challenger.features].to_numpy(np.float32)
    )
    out = {"challenger": evaluate_scores(y, challenger_scores, challenger.threshold, families)}
    out["champion"] = None
    if champion_dir is not None and champion_threshold is not None:
        out["champion"] = evaluate_scores(
            y, score_with_onnx(champion_dir, holdout), champion_threshold, families
        )
    return out


# ============= Retraining run =============


@dataclass
class RetrainDeps:
    """External dependencies of a retraining run, injectable for tests."""

    registry: Any
    build_window: Any
    validator: Any
    set_rollout_mode: Any
    report_status: Any
    model_config: dict[str, Any]
    retrain_config: dict[str, Any]
    features: list[str]


def run_retraining(params: dict[str, Any], deps: RetrainDeps) -> dict[str, Any]:
    """
    Execute one retraining run end to end.

    Steps: window build, GX validation (reject on failure, never loosened), train, ONNX export
    with parity check, champion versus challenger on the same holdout, registration with tags and
    the `challenger` alias, rollout mode switched to `shadow`.

    Parameters
    ----------
    params : dict
        `dataset`, `trigger_id`, `window_id`, `window_end_ms`.
    deps : RetrainDeps
        Injected dependencies.

    Returns
    -------
    summary : dict
        Final `status` and supporting numbers.
    """
    dataset, trigger_id = params["dataset"], params["trigger_id"]
    cfg = deps.retrain_config
    frame = deps.build_window(int(params["window_end_ms"]), int(cfg["train_window_labeled"]))
    summary: dict[str, Any] = {
        "trigger_id": trigger_id,
        "window_id": params["window_id"],
        "window_rows": len(frame),
    }

    report = deps.validator.validate(frame)
    if not report.success:
        summary.update(status=STATUS_REJECTED_DQ, failed=report.failed[:50])
        deps.report_status(trigger_id, STATUS_REJECTED_DQ)
        logger.warning(
            "retrain_rejected_data_quality", trigger_id=trigger_id, failed=report.failed[:10]
        )
        return summary

    from driftguard.training.window import split_window

    splits = split_window(frame, float(cfg["holdout_fraction"]), float(cfg["validation_fraction"]))
    min_pos = int(cfg["min_holdout_positives"])
    if splits.validation["label"].sum() == 0 or splits.holdout["label"].sum() < min_pos:
        summary.update(
            status=STATUS_INSUFFICIENT,
            validation_positive=int(splits.validation["label"].sum()),
            holdout_positive=int(splits.holdout["label"].sum()),
        )
        deps.report_status(trigger_id, STATUS_INSUFFICIENT)
        return summary

    trained = train_and_package(
        splits, deps.features, deps.model_config, int(cfg["reference_max_rows"])
    )
    name = model_name(dataset)
    champion = deps.registry.get_alias(name, CHAMPION)
    with tempfile.TemporaryDirectory() as tmp:
        champion_dir = deps.registry.download(champion, Path(tmp)) if champion else None
        comparison = compare_on_holdout(
            splits.holdout, trained, champion_dir, champion.threshold if champion else None
        )
    challenger_metrics = comparison["challenger"]
    tags = {
        "trigger_id": trigger_id,
        "window_end": int(params["window_end_ms"]),
        "train_rows": len(splits.fit) + len(splits.validation),
        "holdout_rows": len(splits.holdout),
        "data_hash": data_hash(frame),
        "threshold": trained.threshold,
        "parity_max_abs_diff": trained.parity_max_abs_diff,
        "holdout_comparison": json.dumps(
            {
                k: (v and {m: v[m] for m in ("pr_auc", "roc_auc", "precision", "recall")})
                for k, v in comparison.items()
            }
        ),
    }
    metrics = {
        f"holdout_{k}": challenger_metrics[k]
        for k in ("pr_auc", "roc_auc", "precision", "recall")
        if challenger_metrics[k] is not None
    }
    version = deps.registry.register(dataset, trained.artifacts(), tags, metrics, trained.params)
    deps.registry.set_alias(name, CHALLENGER, version)
    deps.set_rollout_mode(dataset, "shadow")
    deps.report_status(trigger_id, STATUS_REGISTERED)
    summary.update(
        status=STATUS_REGISTERED,
        challenger_version=version,
        champion_version=champion.version if champion else None,
        holdout=comparison,
        threshold=trained.threshold,
    )
    logger.info("challenger_registered", trigger_id=trigger_id, version=version)
    return summary
