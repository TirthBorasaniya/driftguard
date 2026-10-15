"""Tests for the Kubernetes layer: chart files, rendered manifests, and E3b analysis."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from driftguard.benchmark.k8s_autoscaling import analyze, job_manifest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
CHART = ROOT / "deploy" / "k8s" / "charts" / "driftguard"
HELM = shutil.which("helm") or (
    str(ROOT / ".cache/bin/helm") if (ROOT / ".cache/bin/helm").exists() else None
)
needs_helm = pytest.mark.skipif(HELM is None, reason="helm not installed")


def render(*extra: str) -> list[dict]:
    """Render the chart with the kind values and return every document."""
    out = subprocess.run(
        [
            HELM,
            "template",
            "driftguard",
            str(CHART),
            "-f",
            str(ROOT / "deploy/k8s/values-kind.yaml"),
            "--set",
            "secrets.adminToken=x",
            "--set",
            "secrets.ipHashSalt=y",
            *extra,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [doc for doc in yaml.safe_load_all(out) if doc]


def test_chart_files_in_sync():
    result = subprocess.run(
        [sys.executable, "scripts/sync_chart_files.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout


def test_static_manifests():
    keda = yaml.safe_load((ROOT / "deploy/k8s/keda/scaledobject.yaml").read_text())
    assert keda["spec"]["maxReplicaCount"] == 6 and keda["spec"]["minReplicaCount"] == 1
    trigger = keda["spec"]["triggers"][0]
    assert trigger["type"] == "kafka" and trigger["metadata"]["lagThreshold"] == "5000"
    assert trigger["metadata"]["consumerGroup"] == "driftguard-scoring"
    assert keda["spec"]["cooldownPeriod"] == 60
    topics = list(yaml.safe_load_all((ROOT / "deploy/k8s/strimzi/topics.yaml").read_text()))
    partitions = {t["metadata"]["name"]: t["spec"]["partitions"] for t in topics}
    assert partitions["flows.cicids2017.v1"] == 6 and partitions["drift.events"] == 1
    assert partitions["flows.dlq"] == 1 and partitions["predictions.v1"] == 6


@needs_helm
def test_rendered_pods_have_resources_and_serving_is_protected():
    docs = render("--set", "producer.enabled=true")
    workloads = [d for d in docs if d["kind"] in ("Deployment", "Job")]
    assert len(workloads) >= 9
    for doc in workloads:
        for container in doc["spec"]["template"]["spec"]["containers"]:
            assert container["resources"]["requests"], doc["metadata"]["name"]
            assert container["resources"]["limits"], doc["metadata"]["name"]
    serving = next(
        d
        for d in docs
        if d["kind"] == "Deployment" and d["metadata"]["name"] == "driftguard-serving"
    )
    assert serving["spec"]["replicas"] == 2
    pdb = next(d for d in docs if d["kind"] == "PodDisruptionBudget")
    assert pdb["spec"]["minAvailable"] == 1
    assert any(d["kind"] == "HorizontalPodAutoscaler" for d in docs)
    consumer = next(
        d
        for d in docs
        if d["kind"] == "Deployment" and d["metadata"]["name"] == "driftguard-consumer"
    )
    assert "replicas" not in consumer["spec"], "KEDA owns the consumer replica count"
    monitors = {d["metadata"]["name"] for d in docs if d["kind"] == "ServiceMonitor"}
    assert monitors == {
        f"driftguard-{c}" for c in ("serving", "consumer", "monitor", "orchestrator")
    }
    for kind in ("PrometheusRule", "ConfigMap"):
        assert any(d["kind"] == kind for d in docs)


@needs_helm
def test_secrets_are_required():
    result = subprocess.run(
        [HELM, "template", "driftguard", str(CHART)], capture_output=True, text=True
    )
    assert result.returncode != 0 and "secrets.adminToken is required" in result.stderr


@needs_helm
def test_job_manifest_rewrites_command_and_name():
    config = {
        "chart": "deploy/k8s/charts/driftguard",
        "values": "deploy/k8s/values-kind.yaml",
        "namespace": "driftguard",
    }
    job = yaml.safe_load(job_manifest(config, "probe", ["python", "-c"], ["print(1)"]))
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert job["metadata"]["name"] == "probe" and container["args"] == ["print(1)"]
    assert container["command"] == ["python", "-c"]


def test_analyze_scaling_timings():
    phases = [
        {"rate_eps": 250, "duration_s": 100},
        {"rate_eps": 2000, "duration_s": 200},
        {"rate_eps": 250, "duration_s": 100},
    ]
    samples = [
        {"t_s": 0, "replicas": 1, "lag": 0},
        {"t_s": 90, "replicas": 1, "lag": 10},
        {"t_s": 100, "replicas": 1, "lag": 2000},
        {"t_s": 130, "replicas": 1, "lag": 40000},
        {"t_s": 160, "replicas": 3, "lag": 90000},
        {"t_s": 250, "replicas": 6, "lag": 30000},
        {"t_s": 300, "replicas": 6, "lag": 20000},
        {"t_s": 340, "replicas": 6, "lag": 4000},
        {"t_s": 420, "replicas": 3, "lag": 100},
        {"t_s": 500, "replicas": 1, "lag": 0},
    ]
    result = analyze(samples, phases, threshold=5000)
    assert result["time_to_first_new_pod_ready_s"] == 60
    assert result["peak_lag"] == 90000 and result["max_replicas"] == 6
    assert result["time_to_drain_below_threshold_s"] == 40
    assert result["scale_in_after_load_drop_s"] == 200
