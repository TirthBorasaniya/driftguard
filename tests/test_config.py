"""Tests for configuration module."""

from src.config import (
    CATEGORICAL_COLS,
    CICIDS_COLUMN_MAP,
    ENTITY_COL,
    FEATURE_COLS,
    LGBM_PARAMS,
    NUMERIC_COLS,
    PROJECT_ROOT,
    TARGET_COL,
    settings,
)


def test_feature_counts():
    assert len(FEATURE_COLS) == 10
    assert len(CATEGORICAL_COLS) == 0
    assert NUMERIC_COLS == FEATURE_COLS


def test_no_duplicate_features():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_domain_identifiers():
    assert TARGET_COL == "label_binary"
    assert ENTITY_COL == "src_ip"


def test_kafka_topics():
    assert settings.kafka_topic == "network_flows"
    assert settings.kafka_dlq_topic == "network_flows.dlq"


def test_mlflow_experiment_name():
    assert settings.mlflow_experiment_name == "network_anomaly_detection"


def test_cicids_column_map_targets_schema_fields():
    assert CICIDS_COLUMN_MAP["Flow Duration"] == "flow_duration"
    assert CICIDS_COLUMN_MAP["Source IP"] == "src_ip"
    assert CICIDS_COLUMN_MAP["Label"] == "label"


def test_lgbm_params_valid():
    assert LGBM_PARAMS["objective"] == "binary"
    assert "average_precision" in LGBM_PARAMS["metric"]
    assert 0 < LGBM_PARAMS["learning_rate"] < 1


def test_project_root_exists():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "src").exists()


def test_settings_defaults():
    assert settings.drift_min_window == 500
    assert settings.healing_mode in ("AUTO", "SHADOW", "ALERT_ONLY")
