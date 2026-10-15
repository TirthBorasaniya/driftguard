"""Tests for metric names (SPEC 8), alert rules, dashboards, and load-test helpers."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from prometheus_client.metrics import MetricWrapperBase

from driftguard.benchmark.loadtest import percentile_summary, saturated
from driftguard.common import metrics

pytestmark = pytest.mark.unit

SPEC_METRICS = {
    "events_consumed_total": ("counter", {"dataset"}),
    "events_dlq_total": ("counter", {"dataset", "error_type"}),
    "duplicates_total": ("counter", {"dataset"}),
    "consumer_lag": ("gauge", {"dataset", "partition"}),
    "scoring_latency_seconds": ("histogram", {"dataset", "alias"}),
    "psi_max": ("gauge", {"dataset"}),
    "psi_share": ("gauge", {"dataset"}),
    "drift_share": ("gauge", {"dataset"}),
    "score_psi": ("gauge", {"dataset"}),
    "adwin_detections_total": ("counter", {"dataset"}),
    "kswin_detections_total": ("counter", {"dataset"}),
    "drift_events_total": ("counter", {"dataset", "severity"}),
    "retraining_runs_total": ("counter", {"dataset", "status"}),
    "promotions_total": ("counter", {"dataset", "result"}),
    "model_info": ("gauge", {"dataset", "alias", "version"}),
    "shadow_agreement_ratio": ("gauge", {"dataset"}),
}
SPEC_ALERTS = {
    "ScoringLatencyP95Breach",
    "ScoringLatencyP99Breach",
    "ConsumerLagHigh",
    "DLQRateHigh",
    "DriftDetected",
    "RetrainingFailed",
    "DataQualityFailed",
    "CanaryRolledBack",
    "ScoringUnavailable",
}


def registered_metrics() -> dict:
    """Map metric family name (without counter suffix) to (type, labels) for DriftGuard metrics."""
    found = {}
    for value in vars(metrics).values():
        if isinstance(value, MetricWrapperBase):
            found[value._name] = (value._type, set(value._labelnames))  # noqa: SLF001
    return found


def load_yaml(path: str) -> dict:
    """Parse a YAML file from the repository."""
    return yaml.safe_load(Path(path).read_text())


def test_spec_metrics_exist_with_labels():
    found = registered_metrics()
    for name, (kind, labels) in SPEC_METRICS.items():
        family = f"driftguard_{name}".removesuffix("_total")
        assert family in found, name
        assert found[family][0] == kind, name
        assert found[family][1] == labels, name


def test_alert_rules_present_and_reference_existing_metrics():
    rules = load_yaml("prometheus/alerts.yml")
    alerts = {rule["alert"]: rule["expr"] for group in rules["groups"] for rule in group["rules"]}
    assert set(alerts) == SPEC_ALERTS
    families = set(registered_metrics())
    for name, expr in alerts.items():
        for metric in re.findall(r"driftguard_[a-z_]+", expr):
            base = re.sub(r"(_bucket|_total)$", "", metric)
            assert base in families, f"{name} references unknown {metric}"
    assert "50000" in alerts["ConsumerLagHigh"] and "0.01" in alerts["DLQRateHigh"]


def test_dashboards_in_sync_and_reference_existing_metrics():
    result = subprocess.run(
        [sys.executable, "scripts/build_dashboards.py", "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout
    families = set(registered_metrics())
    for name in ("pipeline_health", "drift_monitor", "model_lifecycle"):
        board = json.loads(Path(f"grafana/dashboards/{name}.json").read_text())
        assert board["panels"]
        for panel in board["panels"]:
            for target in panel["targets"]:
                for metric in re.findall(r"driftguard_[a-z_]+", target["expr"]):
                    assert re.sub(r"(_bucket|_total)$", "", metric) in families, metric


def test_prometheus_scrapes_every_service():
    config = load_yaml("prometheus/prometheus.yml")
    jobs = {job["job_name"] for job in config["scrape_configs"]}
    assert jobs == {"serving", "consumer", "monitor", "orchestrator"}


def test_saturation_rule_and_summary():
    assert saturated([0, 1000, 3000, 6000, 9000], rate_eps=1000, backlog_s=5)
    assert not saturated([0, 100, 50, 80, 60], rate_eps=1000, backlog_s=5)
    assert not saturated([0, 10, 20, 30], rate_eps=1000, backlog_s=5)
    assert percentile_summary([]) == {"max": None, "p95": None}
    assert percentile_summary([1.0, 2.0, 3.0])["max"] == 3.0


def test_compose_has_twelve_long_running_services_and_loopback_ports():
    compose = load_yaml("docker-compose.yml")
    services = {name for name, svc in compose["services"].items() if "profiles" not in svc}
    assert services == {
        "kafka",
        "schema-registry",
        "redis",
        "mlflow",
        "prefect-server",
        "prefect-worker",
        "serving",
        "consumer",
        "monitor",
        "orchestrator",
        "prometheus",
        "grafana",
    }
    for svc in compose["services"].values():
        for port in svc.get("ports", []):
            assert str(port).startswith("127.0.0.1:")
        image = svc.get("image", "")
        assert not image.endswith(":latest") and (":" in image or "build" in svc)


def test_step_quantile_query_covers_the_whole_step_not_a_window():
    """Regression (M6): E3 used to report the median of per-30 s p99 samples."""
    from driftguard.benchmark.loadtest import step_quantile_query

    query = step_quantile_query(
        0.99, "driftguard_scoring_roundtrip_seconds", "consumer", "cicids2017", 305
    )
    assert "increase(driftguard_scoring_roundtrip_seconds_bucket" in query
    assert "[305s]" in query and 'job="consumer"' in query and "histogram_quantile(0.99" in query


def test_reset_step_state_returns_the_stack_to_idle():
    """Regression (M7): a forced retraining left shadow mode and a challenger for later steps."""
    import fakeredis

    from driftguard.benchmark.loadtest import reset_step_state
    from driftguard.common.mlflow_client import CHALLENGER, CHAMPION, model_name

    from .fakes import FakeRegistry

    registry = FakeRegistry()
    registry.register("cicids2017", {}, {"threshold": 0.5})
    registry.register("cicids2017", {}, {"threshold": 0.5})
    name = model_name("cicids2017")
    registry.set_alias(name, CHAMPION, "1")
    registry.set_alias(name, CHALLENGER, "2")
    redis = fakeredis.FakeRedis(decode_responses=True)
    redis.set("dg:rollout:cicids2017", "shadow")
    for key in ("dg:seen:a", "dg:seen:b", "dg:sf:z:src:x", "dg:sf:agg:y", "dg:retrain:paused:x"):
        redis.set(key, "1")
    reset_step_state(redis, registry, "cicids2017")
    assert registry.get_alias(name, CHALLENGER) is None
    assert registry.get_alias(name, CHAMPION).version == "1"
    assert redis.get("dg:rollout:cicids2017") == "champion_only"
    assert sorted(redis.keys("*")) == ["dg:retrain:paused:x", "dg:rollout:cicids2017"]


def test_e3_renderer_uses_step_level_latency_fields():
    from driftguard.common.render import render_e3_load

    latency = {
        "roundtrip_p50": 1.0,
        "roundtrip_p95": 2.0,
        "roundtrip_p99": 3.0,
        "model_p50": 0.5,
        "model_p95": 0.7,
        "model_p99": 0.9,
        "worst_30s_model_p99": 1.4,
    }
    step = {
        "target_rate_eps": 100,
        "achieved_consumption_rate_eps": 99.0,
        "lag": {"max": 5.0, "p95": 4.0},
        "latency_ms": latency,
        "dlq_rate": 0.0,
        "redis_peak_used_memory_bytes": 2 * 1024**2,
        "saturated": False,
    }
    doc = {
        "metrics": {
            "partitions": 6,
            "runs": {"stream_features=on/idle": {"steps": [step], "saturation_rate_eps": None}},
        }
    }
    text = render_e3_load(doc)
    assert "Round-trip p99 ms" in text and "Worst 30 s model p99 ms" in text and "1.4" in text
