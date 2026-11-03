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


def test_e2_renderer_shows_the_data_quality_gate_table():
    from driftguard.common.render import render_e2_drift_detection

    detector = {
        "median_delay_windows": 1.0,
        "iqr_delay_windows": [0.0, 2.0],
        "miss_rate": 0.0,
        "false_alarms_per_100_windows": 0.0,
    }
    detectors = {name: detector for name in ("psi_max", "policy")}
    result = {
        "detectors": detectors,
        "gx": {
            "failures_per_100_windows_before_onset": 0.0,
            "share_of_windows_blocked_after_onset": 0.25,
        },
    }
    doc = {
        "metrics": {
            "datasets": {
                "cicids2017": {"results": {"label_coverage=1.0": {"covariate:factor=1.5": result}}}
            }
        }
    }
    text = render_e2_drift_detection(doc)
    assert "Data-quality gate" in text and "Share of post-onset windows blocked" in text
    assert "0.25" in text


def test_headline_lists_every_guarantee_with_a_results_file(tmp_path):
    """Regression: several rows were silently dropped by a failed text replacement."""
    from driftguard.common.render import HEADLINE_ROWS, render_headline
    from driftguard.common.results import write_results

    names = {row[1] for row in HEADLINE_ROWS}
    assert names == {
        "e1_model_quality",
        "e2_drift_detection",
        "e3_load",
        "e3b_k8s_autoscaling",
        "e4_adaptation",
        "e5_gate",
    }
    assert len(HEADLINE_ROWS) == 7
    assert render_headline(tmp_path) == ""
    metrics = {"challengers": {"better": {"decision": "promote"}}, "all_match_expected": True}
    write_results("e5_gate", metrics, None, results_dir=tmp_path)
    text = render_headline(tmp_path)
    assert text.startswith("## Headline results") and "results/e5_gate.json" in text
    assert text.count("`results/") == 1 and "results/e3_load.json" not in text
    assert "not yet run" not in text.lower()


def _rates():
    return {
        "pr_auc": 0.9,
        "roc_auc": 0.95,
        "precision": 0.8,
        "recall": 0.7,
        "n": 10,
        "n_positive": 3,
        "oracle_precision_at_recall_0_99": 0.6,
    }


def _protocol():
    return {
        "threshold": 0.9,
        "test": {**_rates(), "per_family_recall": {"DDoS": {"recall": 0.5, "n": 4}}},
        "test_at_validation_recall_0_99": {"threshold": 0.8, "precision": 0.7, "recall": 0.99},
    }


def test_every_renderer_and_headline_extractor_works_on_a_realistic_document():
    """Regression: E3b and E4 renderers and several headline rows were never exercised."""
    from driftguard.common.render import HEADLINE_ROWS, RENDERERS

    detector = {
        "median_delay_windows": 1.0,
        "iqr_delay_windows": [0.0, 2.0],
        "miss_rate": 0.0,
        "false_alarms_per_100_windows": 0.0,
        "detected": 2,
        "runs": 2,
    }
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
    decision = {
        "decision": "promote",
        "expected": "promote",
        "delta_pr_auc": 0.5,
        "ci_low": 0.4,
        "ci_high": 0.6,
        "margin": 0.01,
        "n_resamples": 1000,
        "matches_expected": True,
    }
    docs = {
        "e1_model_quality": {
            "protocols": {"time_ordered": _protocol(), "within_day": _protocol()},
            "ablations": {
                "o_drop_dst_port": _protocol(),
                "attempted_policy": {"benign": _protocol()},
                "stream_features": {
                    "cicflowmeter_only": _protocol(),
                    "cicflowmeter_plus_behavioral": _protocol(),
                },
            },
        },
        "e2_drift_detection": {
            "datasets": {
                "cicids2017": {
                    "results": {
                        "label_coverage=1.0": {
                            "covariate:factor=1.5": {
                                "detectors": {n: detector for n in ("psi_max", "policy")},
                                "gx": {
                                    "failures_per_100_windows_before_onset": 0.0,
                                    "share_of_windows_blocked_after_onset": 0.0,
                                },
                            }
                        }
                    }
                }
            }
        },
        "e3_load": {
            "partitions": 6,
            "runs": {
                "stream_features=on/idle": {"steps": [step], "saturation_rate_eps": None},
                "stream_features=on/retraining": {"steps": [step], "saturation_rate_eps": None},
            },
        },
        "e3b_k8s_autoscaling": {
            "phases": [{"rate_eps": 250, "duration_s": 300}],
            "scaling": {
                "time_to_first_new_pod_ready_s": 40,
                "peak_lag": 9000,
                "max_replicas": 6,
                "time_to_drain_below_threshold_s": 120,
                "scale_in_after_load_drop_s": 200,
            },
            "duplicate_and_loss_check": {
                "produced_distinct": 10,
                "predicted_distinct": 10,
                "duplicates": 0,
                "lost": 0,
            },
            "figure": "results/figures/e3b_replicas.png",
        },
        "e4_adaptation": {
            "kyoto": {
                "modes": {
                    "static": {
                        "pr_auc_per_year": {"2008": {"pr_auc": 0.5}},
                        "lifecycle": {
                            "drift_events": 0,
                            "promoted_to_canary": 0,
                            "promotions": 0,
                            "declines": 0,
                            "rollbacks": 0,
                        },
                    },
                    "adaptive": {
                        "pr_auc_per_year": {"2008": {"pr_auc": 0.7}},
                        "lifecycle": {
                            "drift_events": 2,
                            "promoted_to_canary": 1,
                            "promotions": 1,
                            "declines": 0,
                            "rollbacks": 0,
                        },
                    },
                },
                "figure": "results/f.png",
            },
            "cicids2017": {
                "modes": {
                    "adaptive": {
                        "lifecycle": {
                            "drift_events": 1,
                            "promoted_to_canary": 0,
                            "promotions": 0,
                            "declines": 1,
                            "rollbacks": 0,
                        }
                    }
                },
                "figure": "results/c.png",
            },
        },
        "e5_gate": {
            "shadow_events": 3000,
            "shadow_positive": 300,
            "all_match_expected": True,
            "challengers": {
                "better": decision,
                "equal": {**decision, "decision": "decline", "expected": "decline"},
            },
        },
        "onnx_latency": {
            "rows": [
                {
                    "batch_size": 1,
                    "native": {"p50_ms": 0.2, "p95_ms": 0.3},
                    "onnx": {"p50_ms": 0.01, "p95_ms": 0.02},
                    "speedup_p50": 20.0,
                }
            ]
        },
    }
    for name, payload in docs.items():
        doc = {
            "run_id": f"t_{name}",
            "git_sha": "abc",
            "environment": {"hardware": "h"},
            "metrics": payload,
            "notes": "",
        }
        _, renderer = RENDERERS[name]
        assert renderer(doc)
        for _claim, row_name, _make, extractor in HEADLINE_ROWS:
            if row_name == name:
                assert extractor(doc)
