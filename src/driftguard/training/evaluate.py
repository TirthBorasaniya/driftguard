"""Evaluation metrics: PR-AUC, ROC-AUC, thresholded precision and recall, per-family recall."""

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

# ============= Metrics =============


def precision_at_recall(y: np.ndarray, scores: np.ndarray, recall_level: float) -> float | None:
    """
    Return the best precision achievable at recall >= `recall_level` (an oracle value).

    The threshold is chosen using the same labels the precision is measured on, so this is an
    upper bound on what a deployed threshold can reach, not an operating point.

    Returns
    -------
    precision : float or None
        None when there are no positives.
    """
    if int(np.asarray(y).sum()) == 0:
        return None
    precision, recall, _ = precision_recall_curve(y, scores)
    eligible = precision[recall >= recall_level]
    return float(eligible.max()) if eligible.size else None


def per_family_recall(
    y: np.ndarray, decisions: np.ndarray, families: np.ndarray
) -> dict[str, dict[str, float | int]]:
    """
    Recall at the stored threshold for each attack family.

    Parameters
    ----------
    y : np.ndarray
        Binary labels, shape (n,).
    decisions : np.ndarray
        Binary decisions at the threshold, shape (n,).
    families : np.ndarray
        Family name per row, shape (n,); only positive rows are counted.

    Returns
    -------
    recall_per_family : dict
        Maps family to `{"recall": float, "n": int}`.
    """
    y, decisions, families = np.asarray(y), np.asarray(decisions), np.asarray(families)
    out = {}
    for family in sorted(set(families[y == 1].tolist())):
        mask = (families == family) & (y == 1)
        out[str(family)] = {"recall": float(decisions[mask].mean()), "n": int(mask.sum())}
    return out


def evaluate_scores(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    families: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute the SPEC 6.2 step 5 metrics for one model on one dataset.

    Parameters
    ----------
    y : np.ndarray
        Binary labels, shape (n,).
    scores : np.ndarray
        Model scores, shape (n,).
    threshold : float
        Stored decision threshold.
    families : np.ndarray or None
        Family per row for per-family recall.

    Returns
    -------
    metrics : dict
        `pr_auc`, `roc_auc`, `precision`, `recall`, `positive_rate`, `n`, `n_positive`,
        `oracle_precision_at_recall_0_99` (threshold chosen on these labels), and
        `per_family_recall` when families are given.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    decisions = (scores >= threshold).astype(int)
    n_pos = int(y.sum())
    tp = int(((decisions == 1) & (y == 1)).sum())
    predicted_pos = int(decisions.sum())
    has_both = 0 < n_pos < len(y)
    metrics: dict[str, Any] = {
        "n": len(y),
        "n_positive": n_pos,
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y, scores)) if n_pos else None,
        "roc_auc": float(roc_auc_score(y, scores)) if has_both else None,
        "precision": tp / predicted_pos if predicted_pos else None,
        "recall": tp / n_pos if n_pos else None,
        "positive_rate": predicted_pos / len(y) if len(y) else None,
        "oracle_precision_at_recall_0_99": precision_at_recall(y, scores, 0.99),
    }
    if families is not None:
        metrics["per_family_recall"] = per_family_recall(y, decisions, families)
    return metrics
