"""Tests for configuration module."""

from src.config import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    LGBM_PARAMS,
    NUMERIC_COLS,
    PROJECT_ROOT,
    settings,
)


def test_feature_counts():
    assert len(NUMERIC_COLS) > 0
    assert len(CATEGORICAL_COLS) == 7
    assert len(FEATURE_COLS) == len(NUMERIC_COLS) + len(CATEGORICAL_COLS)


def test_no_duplicate_features():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_no_overlap_numeric_categorical():
    overlap = set(NUMERIC_COLS) & set(CATEGORICAL_COLS)
    assert len(overlap) == 0, f"Overlap found: {overlap}"


def test_lgbm_params_valid():
    assert LGBM_PARAMS["objective"] == "binary"
    assert LGBM_PARAMS["scale_pos_weight"] > 1
    assert 0 < LGBM_PARAMS["learning_rate"] < 1


def test_project_root_exists():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "src").exists()


def test_settings_defaults():
    assert settings.kafka_topic == "transactions"
    assert settings.drift_min_window == 500
    assert settings.healing_mode in ("AUTO", "SHADOW", "ALERT_ONLY")
