"""Serving threshold calibration: select the operating point that meets a recall floor."""

import numpy as np
from sklearn.metrics import precision_recall_curve


def calibrate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    recall_target: float,
) -> float:
    """
    Find the classification threshold that achieves the target recall.

    Recall decreases monotonically as the threshold rises, so any threshold at
    or below a boundary value satisfies the recall floor. This returns the
    highest such threshold, which meets the recall floor while maximizing
    precision (minimizing false alarms). For network anomaly detection a missed
    attack is more costly than a false alarm, so the recall floor is enforced.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive (attack) class.
    recall_target : float
        Minimum recall to achieve at the returned threshold.

    Returns
    -------
    threshold : float
        Classification threshold calibrated to recall_target.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # thresholds has len(precisions) - 1 elements; recalls[:-1] aligns with it
    candidate_list = [
        float(t) for t, r in zip(thresholds, recalls[:-1]) if r >= recall_target
    ]

    if candidate_list:
        return max(candidate_list)

    # target recall unreachable; fall back to the most permissive threshold
    return float(thresholds.min()) if len(thresholds) else 0.0
