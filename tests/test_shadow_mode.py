"""Tests for shadow mode champion-challenger scoring (infra improvement 6)."""

import numpy as np

from src.config import FEATURE_COLS
from src.serving.shadow_mode import (
    SHADOW_DIVERGENCE_THRESHOLD,
    SHADOW_MODE_MIN_EVENTS,
    evaluate_shadow_divergence,
    log_shadow_prediction,
    score_shadow_mode,
)


class FakeModel:
    """Stand-in for an LGBMClassifier that always predicts a fixed probability."""

    def __init__(self, positive_proba: float):
        self.positive_proba = positive_proba

    def predict_proba(self, features):
        n = len(features)
        return np.array([[1 - self.positive_proba, self.positive_proba]] * n)


def _feature_dict():
    return {col: 1.0 for col in FEATURE_COLS}


def test_score_shadow_mode_returns_both_predictions():
    champion = FakeModel(0.1)
    challenger = FakeModel(0.1)
    champion_pred, challenger_pred = score_shadow_mode(champion, challenger, _feature_dict())
    assert champion_pred == 0.1
    assert challenger_pred == 0.1


def test_score_shadow_mode_does_not_raise_on_disagreement():
    champion = FakeModel(0.1)
    challenger = FakeModel(0.9)
    champion_pred, challenger_pred = score_shadow_mode(champion, challenger, _feature_dict())
    assert champion_pred == 0.1
    assert challenger_pred == 0.9


def test_evaluate_shadow_divergence_below_min_events_rejects(tmp_path):
    log_path = tmp_path / "shadow_log.csv"
    for _ in range(10):
        log_shadow_prediction(str(log_path), 0.1, 0.9, False, True)

    approved = evaluate_shadow_divergence(str(log_path), SHADOW_MODE_MIN_EVENTS, SHADOW_DIVERGENCE_THRESHOLD)
    assert approved is False


def test_evaluate_shadow_divergence_approves_low_disagreement(tmp_path):
    log_path = tmp_path / "shadow_log.csv"
    for _ in range(SHADOW_MODE_MIN_EVENTS):
        log_shadow_prediction(str(log_path), 0.1, 0.1, False, False)

    approved = evaluate_shadow_divergence(str(log_path), SHADOW_MODE_MIN_EVENTS, SHADOW_DIVERGENCE_THRESHOLD)
    assert approved is True


def test_evaluate_shadow_divergence_rejects_high_disagreement(tmp_path):
    log_path = tmp_path / "shadow_log.csv"
    for _ in range(SHADOW_MODE_MIN_EVENTS):
        log_shadow_prediction(str(log_path), 0.1, 0.9, False, True)

    approved = evaluate_shadow_divergence(str(log_path), SHADOW_MODE_MIN_EVENTS, SHADOW_DIVERGENCE_THRESHOLD)
    assert approved is False
