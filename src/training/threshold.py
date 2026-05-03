"""F2 threshold optimization: find decision threshold that maximizes F2 on Precision-Recall curve."""

import numpy as np
from sklearn.metrics import precision_recall_curve


def find_f2_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Find decision threshold that maximizes F2 score on the Precision-Recall curve.

    F2 weights recall twice as heavily as precision, which is the correct
    operating point for fraud detection: missing a fraud (false negative) is
    more costly than a false alarm (false positive).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_proba : np.ndarray
        Predicted fraud probabilities.

    Returns
    -------
    threshold : float
        Optimal decision threshold in [0, 1].
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # F2 = (1 + 2^2) * P * R / (2^2 * P + R) = 5*P*R / (4*P + R)
    denom = 4 * precisions + recalls + 1e-9
    f2_scores = (5 * precisions * recalls) / denom

    # thresholds has len(precisions) - 1 elements
    best_idx = int(np.argmax(f2_scores[:-1]))
    return float(thresholds[best_idx])
