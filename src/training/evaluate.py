"""Model evaluation: AUC-PR, AUC-ROC, F2-score, precision, and recall at a given threshold."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
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
        auc_pr, auc_roc, f2_score, precision, recall at threshold.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "auc_pr": float(average_precision_score(y_test, y_proba)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
        "f2_score": float(fbeta_score(y_test, y_pred, beta=2, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
