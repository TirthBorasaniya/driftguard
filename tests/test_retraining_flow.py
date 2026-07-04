"""Unit tests for retraining business logic (no Kafka, Redis, or data files required)."""

import json

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.training.evaluate import evaluate_model

# ============= Champion promotion logic (PR-AUC margin 0.005) =============


def _import_train():
    """Import the training module, skipping if its heavy deps are absent."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("mlflow")
    import src.training.train as train

    return train


def test_should_promote_no_existing_champion():
    train = _import_train()
    assert train.should_promote({"auc_pr": 0.5}, None) is True


def test_should_promote_improvement_above_margin():
    train = _import_train()
    # 0.02 > 0.005 margin
    assert train.should_promote({"auc_pr": 0.82}, {"auc_pr": 0.80}) is True


def test_should_promote_improvement_below_margin():
    train = _import_train()
    # 0.003 < 0.005 margin
    assert train.should_promote({"auc_pr": 0.803}, {"auc_pr": 0.80}) is False


def test_should_promote_regression_does_not_promote():
    train = _import_train()
    assert train.should_promote({"auc_pr": 0.70}, {"auc_pr": 0.80}) is False


def test_promotion_margin_constant():
    train = _import_train()
    assert train.PROMOTION_PRAUC_MARGIN == pytest.approx(0.005)
    assert train.PROMOTION_METRIC == "pr_auc"
    assert train.SERVING_RECALL_TARGET == pytest.approx(0.95)


# ============= Champion metrics persistence =============


def test_load_champion_metrics_missing_file(tmp_path):
    train = _import_train()
    import src.config as cfg

    original = cfg.CHAMPION_METRICS_PATH
    cfg.CHAMPION_METRICS_PATH = tmp_path / "missing.json"
    try:
        assert train.load_champion_metrics() is None
    finally:
        cfg.CHAMPION_METRICS_PATH = original


def test_load_champion_metrics_reads_json(tmp_path):
    train = _import_train()
    import src.config as cfg

    original = cfg.CHAMPION_METRICS_PATH
    path = tmp_path / "champion_metrics.json"
    path.write_text(json.dumps({"pr_auc": 0.75, "auc_pr": 0.75, "threshold": 0.42, "version": 3}))
    cfg.CHAMPION_METRICS_PATH = path
    try:
        result = train.load_champion_metrics()
        assert result is not None
        assert result["auc_pr"] == pytest.approx(0.75)
        assert result["version"] == 3
    finally:
        cfg.CHAMPION_METRICS_PATH = original


# ============= Model evaluation (PR-AUC primary) =============


def _fit_dummy(seed, n_pos):
    np.random.seed(seed)
    X = np.random.randn(300, 5)
    y = np.array([0] * (300 - n_pos) + [1] * n_pos)
    clf = DummyClassifier(strategy="stratified", random_state=seed)
    clf.fit(X, y)
    return clf, X, y


def test_evaluate_model_returns_required_keys():
    clf, X, y = _fit_dummy(42, 30)
    metrics = evaluate_model(clf, X, y, threshold=0.3)
    required = {"pr_auc", "auc_pr", "auc_roc", "precision", "recall", "threshold"}
    assert required.issubset(metrics.keys())


def test_evaluate_model_no_f2_score():
    clf, X, y = _fit_dummy(42, 30)
    metrics = evaluate_model(clf, X, y, threshold=0.3)
    assert "f2_score" not in metrics


def test_evaluate_model_metrics_in_range():
    clf, X, y = _fit_dummy(42, 30)
    metrics = evaluate_model(clf, X, y, threshold=0.3)
    for key in ("pr_auc", "auc_pr", "auc_roc", "precision", "recall"):
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


def test_evaluate_model_threshold_preserved():
    clf, X, y = _fit_dummy(7, 20)
    metrics = evaluate_model(clf, X, y, threshold=0.42)
    assert metrics["threshold"] == pytest.approx(0.42)
