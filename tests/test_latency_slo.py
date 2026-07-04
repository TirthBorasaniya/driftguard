"""Tests for the latency SLO Prometheus alert rule file (infra improvement 3)."""

from pathlib import Path

import yaml

from src.config import PROJECT_ROOT

ALERTS_PATH = Path(PROJECT_ROOT) / "monitoring" / "prometheus" / "alerts.yml"


def test_alerts_file_parses_as_valid_yaml():
    with open(ALERTS_PATH) as f:
        parsed = yaml.safe_load(f)

    assert "groups" in parsed
    group_names = [g["name"] for g in parsed["groups"]]
    assert "latency_slo" in group_names


def test_alerts_reference_predict_latency_metric():
    with open(ALERTS_PATH) as f:
        parsed = yaml.safe_load(f)

    rules = parsed["groups"][0]["rules"]
    alert_names = [r["alert"] for r in rules]
    assert "PredictLatencyP95Breach" in alert_names
    assert "PredictLatencyP99Breach" in alert_names

    for rule in rules:
        assert "http_request_duration_seconds_bucket" in rule["expr"]
        assert 'handler="/predict"' in rule["expr"]
