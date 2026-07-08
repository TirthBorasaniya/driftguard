"""Unit tests for dynamic scale_pos_weight computation."""

import numpy as np
import pytest


def _import_train():
    """Import the training module, skipping if its heavy deps are absent."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("mlflow")
    import src.training.train as train

    return train


def test_compute_scale_pos_weight_known_ratio():
    train = _import_train()
    y_train = np.array([0] * 100 + [1] * 5)
    assert train.compute_scale_pos_weight(y_train) == pytest.approx(20.0)


def test_compute_scale_pos_weight_balanced_classes():
    train = _import_train()
    y_train = np.array([0] * 50 + [1] * 50)
    assert train.compute_scale_pos_weight(y_train) == pytest.approx(1.0)
