"""Unit tests for retraining flow business logic (no Kafka, Redis, or data files required)."""

import json

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier


# ============= Champion promotion logic =============


def test_should_promote_no_existing_champion():
    """First run with no champion must always promote."""
    from src.training.train import should_promote

    assert should_promote({"auc_pr": 0.5}, None) is True


def test_should_promote_improvement_above_threshold():
    """Improvement > 0.01 must trigger promotion."""
    from src.training.train import should_promote

    assert should_promote({"auc_pr": 0.82}, {"auc_pr": 0.80}) is True


def test_should_promote_improvement_exactly_at_threshold():
    """Improvement == threshold must NOT promote (strictly greater required)."""
    from src.training.train import should_promote

    assert should_promote({"auc_pr": 0.81}, {"auc_pr": 0.80}) is False


def test_should_promote_improvement_below_threshold():
    """Marginal improvement must not promote."""
    from src.training.train import should_promote

    assert should_promote({"auc_pr": 0.8005}, {"auc_pr": 0.800}) is False


def test_should_promote_regression_does_not_promote():
    """Regression (negative improvement) must not promote."""
    from src.training.train import should_promote

    assert should_promote({"auc_pr": 0.70}, {"auc_pr": 0.80}) is False


# ============= Champion metrics persistence =============


def test_load_champion_metrics_missing_file(tmp_path):
    """load_champion_metrics must return None when the JSON file is absent."""
    import src.config as cfg
    from src.training.train import load_champion_metrics

    original = cfg.CHAMPION_METRICS_PATH
    cfg.CHAMPION_METRICS_PATH = tmp_path / "missing.json"
    try:
        result = load_champion_metrics()
        assert result is None
    finally:
        cfg.CHAMPION_METRICS_PATH = original


def test_load_champion_metrics_reads_json(tmp_path):
    """load_champion_metrics must return the dict stored in champion_metrics.json."""
    import src.config as cfg
    from src.training.train import load_champion_metrics

    original = cfg.CHAMPION_METRICS_PATH
    path = tmp_path / "champion_metrics.json"
    path.write_text(json.dumps({"auc_pr": 0.75, "threshold": 0.31, "version": 3}))
    cfg.CHAMPION_METRICS_PATH = path
    try:
        result = load_champion_metrics()
        assert result is not None
        assert result["auc_pr"] == pytest.approx(0.75)
        assert result["version"] == 3
    finally:
        cfg.CHAMPION_METRICS_PATH = original


# ============= Threshold optimization =============


def test_f2_threshold_in_unit_interval():
    """F2-optimized threshold must fall in (0, 1)."""
    from src.training.threshold import find_f2_threshold

    np.random.seed(0)
    y_true = np.array([0] * 950 + [1] * 50)
    y_proba = np.random.beta(1, 5, 1000)
    y_proba[950:] = np.random.beta(5, 1, 50)
    threshold = find_f2_threshold(y_true, y_proba)
    assert 0.0 < threshold < 1.0


def test_f2_threshold_higher_than_default_for_imbalanced():
    """For heavily imbalanced data, F2 threshold should not be 0.5 by default."""
    from src.training.threshold import find_f2_threshold

    np.random.seed(1)
    y_true = np.array([0] * 980 + [1] * 20)
    y_proba = np.concatenate([np.random.uniform(0, 0.3, 980), np.random.uniform(0.4, 1.0, 20)])
    threshold = find_f2_threshold(y_true, y_proba)
    assert threshold != 0.5


# ============= Model evaluation =============


def test_evaluate_model_returns_required_keys():
    """evaluate_model must return auc_pr, auc_roc, f2_score, precision, recall, threshold."""
    from src.training.evaluate import evaluate_model

    np.random.seed(42)
    X = np.random.randn(300, 5)
    y = np.array([0] * 270 + [1] * 30)
    clf = DummyClassifier(strategy="stratified", random_state=42)
    clf.fit(X, y)
    metrics = evaluate_model(clf, X, y, threshold=0.3)

    required = {"auc_pr", "auc_roc", "f2_score", "precision", "recall", "threshold"}
    assert required.issubset(metrics.keys())


def test_evaluate_model_metrics_in_range():
    """All evaluate_model metric values must be between 0 and 1."""
    from src.training.evaluate import evaluate_model

    np.random.seed(42)
    X = np.random.randn(300, 5)
    y = np.array([0] * 270 + [1] * 30)
    clf = DummyClassifier(strategy="stratified", random_state=42)
    clf.fit(X, y)
    metrics = evaluate_model(clf, X, y, threshold=0.3)

    for key in ("auc_pr", "auc_roc", "f2_score", "precision", "recall"):
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


def test_evaluate_model_threshold_preserved():
    """evaluate_model must echo the provided threshold back in the output."""
    from src.training.evaluate import evaluate_model

    np.random.seed(7)
    X = np.random.randn(200, 3)
    y = np.array([0] * 180 + [1] * 20)
    clf = DummyClassifier(strategy="stratified", random_state=7)
    clf.fit(X, y)
    metrics = evaluate_model(clf, X, y, threshold=0.42)
    assert metrics["threshold"] == pytest.approx(0.42)
