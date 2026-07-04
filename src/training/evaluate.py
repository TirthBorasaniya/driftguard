"""Model evaluation: PR-AUC (primary), ROC-AUC, precision, and recall at a given threshold."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> dict:
    """
    Evaluate a trained model on held-out test data.

    PR-AUC (average precision) is the primary metric: it summarizes performance
    across all thresholds and is the standard choice for the severe class
    imbalance of network anomaly detection.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Trained model with predict_proba method.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        Ground truth labels.
    threshold : float
        Decision threshold for binary classification.

    Returns
    -------
    metrics : dict
        pr_auc, auc_pr, auc_roc, precision, recall at threshold.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_test, y_proba))

    return {
        # pr_auc is the canonical key; auc_pr retained for promotion-comparison compatibility
        "pr_auc": pr_auc,
        "auc_pr": pr_auc,
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
