"""Tests for training utilities: class weight, threshold selection, ONNX parity, metrics."""

import numpy as np
import pytest

from driftguard.training.evaluate import evaluate_scores, per_family_recall
from driftguard.training.export import onnx_scores, onnx_session, parity_check, to_onnx_bytes
from driftguard.training.train import compute_scale_pos_weight, select_threshold, train_model

pytestmark = pytest.mark.unit

PARAMS = {
    "objective": "binary",
    "n_estimators": 150,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "min_child_samples": 5,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
    "n_jobs": 1,
}


def _data(n=3000, f=8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, f)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.5, size=n) > 1.5).astype(int)
    return x, y


def test_scale_pos_weight():
    assert compute_scale_pos_weight(np.array([0] * 100 + [1] * 5)) == 20.0
    with pytest.raises(ValueError):
        compute_scale_pos_weight(np.zeros(4))


def test_select_threshold_is_strictest_meeting_target():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.65, 0.3, 0.2, 0.1, 0.4, 0.05])
    threshold = select_threshold(y, scores, 0.8)
    recall = ((scores >= threshold) & (y == 1)).sum() / y.sum()
    assert recall >= 0.8
    assert threshold == 0.6
    stricter = np.nextafter(threshold, 1.0)
    assert ((scores >= stricter) & (y == 1)).sum() / y.sum() < 0.8


def test_select_threshold_requires_positives():
    with pytest.raises(ValueError):
        select_threshold(np.zeros(3), np.array([0.1, 0.2, 0.3]), 0.95)


def test_training_is_deterministic_and_onnx_parity_holds():
    x, y = _data()
    model_a, spw = train_model(x, y, PARAMS, seed=42)
    model_b, _ = train_model(x, y, PARAMS, seed=42)
    assert spw == pytest.approx((y == 0).sum() / (y == 1).sum())
    assert np.array_equal(model_a.predict_proba(x), model_b.predict_proba(x))
    session = onnx_session(to_onnx_bytes(model_a, x.shape[1], split=50))
    observed = parity_check(model_a, session, x, n_rows=1000, max_abs_diff=1e-5)
    assert observed <= 1e-5
    assert onnx_scores(session, x[:5]).shape == (5,)


def test_evaluate_scores_and_family_recall():
    y = np.array([0, 1, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    families = np.array(["BENIGN", "DDoS", "Bot", "DDoS", "BENIGN"])
    metrics = evaluate_scores(y, scores, 0.5, families)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["per_family_recall"] == {
        "Bot": {"recall": 0.0, "n": 1},
        "DDoS": {"recall": 1.0, "n": 2},
    }
    assert per_family_recall(y, (scores >= 0.5).astype(int), families)["DDoS"]["n"] == 2
