"""Tests for the recall-calibrated serving threshold (operating point)."""

import numpy as np
import pytest
from sklearn.metrics import recall_score

from src.training.threshold import calibrate_threshold

# mirrors SERVING_RECALL_TARGET in src.training.train (imported without lightgbm here)
RECALL_TARGET = 0.95


@pytest.fixture
def imbalanced_scores():
    """Synthetic imbalanced labels with reasonably separable scores."""
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 950 + [1] * 50)
    y_prob = np.concatenate(
        [rng.uniform(0.0, 0.4, 950), rng.uniform(0.5, 1.0, 50)]
    )
    return y_true, y_prob


def test_calibrate_threshold_meets_recall_target(imbalanced_scores):
    y_true, y_prob = imbalanced_scores
    threshold = calibrate_threshold(y_true, y_prob, RECALL_TARGET)
    achieved = recall_score(y_true, (y_prob >= threshold).astype(int))
    assert achieved >= RECALL_TARGET


def test_calibrate_threshold_in_unit_interval(imbalanced_scores):
    y_true, y_prob = imbalanced_scores
    threshold = calibrate_threshold(y_true, y_prob, RECALL_TARGET)
    assert 0.0 <= threshold <= 1.0


def test_lower_recall_target_allows_higher_threshold(imbalanced_scores):
    y_true, y_prob = imbalanced_scores
    thr_high_recall = calibrate_threshold(y_true, y_prob, 0.99)
    thr_low_recall = calibrate_threshold(y_true, y_prob, 0.80)
    # a stricter recall floor cannot raise the threshold above a looser floor
    assert thr_high_recall <= thr_low_recall
