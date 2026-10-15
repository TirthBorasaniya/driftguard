"""LightGBM training with label-derived class weights and recall-targeted thresholds."""

import math
from typing import Any

import lightgbm as lgb
import numpy as np

# ============= Class weighting =============


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    Compute `n_negative / n_positive` from training labels.

    Parameters
    ----------
    y : np.ndarray
        Binary labels, shape (n,).

    Returns
    -------
    weight : float
        Ratio of negatives to positives.
    """
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("training window has no positive labels")
    return n_neg / n_pos


# ============= Training =============


def train_model(
    x: np.ndarray, y: np.ndarray, params: dict[str, Any], seed: int
) -> tuple[lgb.LGBMClassifier, float]:
    """
    Fit a LightGBM classifier with fixed hyperparameters.

    Parameters
    ----------
    x : np.ndarray
        Features, float32, shape (n, f).
    y : np.ndarray
        Binary labels, shape (n,).
    params : dict
        LightGBM parameters from `configs/model.yaml` (`lightgbm` block).
    seed : int
        Random seed applied to every LightGBM randomness source.

    Returns
    -------
    model : lgb.LGBMClassifier
        Fitted model.
    scale_pos_weight : float
        Class weight used for the fit.
    """
    if len(x) == 0:
        raise ValueError("empty training matrix")
    scale_pos_weight = compute_scale_pos_weight(y)
    model = lgb.LGBMClassifier(**params, random_state=seed, scale_pos_weight=scale_pos_weight)
    model.fit(np.asarray(x, dtype=np.float32), np.asarray(y))
    return model, scale_pos_weight


def predict_scores(model: lgb.LGBMClassifier, x: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities, shape (n,)."""
    return model.predict_proba(np.asarray(x, dtype=np.float32))[:, 1]


# ============= Threshold selection =============


def select_threshold(y: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """
    Choose the decision threshold on a validation slice.

    Scanning thresholds from high to low, recall first reaches `target_recall` at the k-th
    largest positive score with `k = ceil(target_recall * n_positive)`. That threshold is the
    strictest one meeting the target, so it maximizes precision subject to the recall floor.

    Parameters
    ----------
    y : np.ndarray
        Binary validation labels, shape (n,).
    scores : np.ndarray
        Validation scores, shape (n,).
    target_recall : float
        Recall target in (0, 1].

    Returns
    -------
    threshold : float
        Events with `score >= threshold` are classified as attacks.
    """
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")
    positive_scores = np.sort(np.asarray(scores)[np.asarray(y) == 1])[::-1]
    if positive_scores.size == 0:
        raise ValueError("validation slice has no positive labels")
    k = math.ceil(target_recall * positive_scores.size - 1e-12)
    return float(positive_scores[k - 1])
