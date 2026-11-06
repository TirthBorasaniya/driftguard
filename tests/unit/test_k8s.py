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


def test_install_does_not_wait_on_serving_before_bootstrap_runs():
    """Regression (M11): `helm install --wait` blocks on serving readiness, which needs the champion
    created by the bootstrap Job, so a post-install hook plus --wait deadlocks."""
    script = (ROOT / "scripts" / "k8s_up.sh").read_text()
    install = script[
        script.index("helm upgrade --install driftguard") : script.index(
            "kubectl apply -f deploy/k8s/keda"
        )
    ]
    assert "--wait" not in install
    assert "job/driftguard-bootstrap" in script and "--for=condition=complete" in script
    jobs = (CHART / "templates" / "jobs.yaml").read_text()
    assert "helm.sh/hook" not in jobs


# ============= Strimzi operator install =============

DUPLICATED = """\
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata: {name: strimzi-cluster-operator, namespace: driftguard}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata: {name: strimzi-cluster-operator, namespace: driftguard}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata: {name: strimzi-cluster-operator, namespace: strimzi}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata: {name: strimzi-cluster-operator}
---
"""


def test_duplicate_resources_are_found_by_kind_namespace_and_name():
    from driftguard.common.manifests import find_duplicate_resources

    assert find_duplicate_resources(DUPLICATED) == [
        ("RoleBinding", "driftguard", "strimzi-cluster-operator", 2)
    ]
    assert find_duplicate_resources("") == []


def test_duplicate_checker_exit_code(tmp_path):
    bad, good = tmp_path / "bad.yaml", tmp_path / "good.yaml"
    bad.write_text(DUPLICATED)
    good.write_text(DUPLICATED.split("---\n", 2)[2])
    script = str(ROOT / "scripts" / "check_duplicate_resources.py")
    run = lambda path: subprocess.run(  # noqa: E731
        [sys.executable, script, str(path)], capture_output=True, text=True, check=False
    )
    assert run(bad).returncode == 1 and "RoleBinding driftguard/strimzi" in run(bad).stdout
    assert run(good).returncode == 0


def test_strimzi_operator_does_not_watch_its_own_namespace():
    """Regression: listing the operator's namespace in watchNamespaces renders the chart's
    RoleBindings twice and `helm install` fails with "rolebindings ... already exist"."""
    script = (ROOT / "scripts" / "k8s_up.sh").read_text()
    values = {
        line.split("=", 1)[0]: line.split("=", 1)[1]
        for line in script.splitlines()
        if line.startswith(("APP_NAMESPACE=", "STRIMZI_NAMESPACE="))
    }
    assert values["APP_NAMESPACE"] != values["STRIMZI_NAMESPACE"]
    install = script[
        script.index("helm upgrade --install strimzi") : script.index("helm upgrade --install keda")
    ]
    assert '--namespace "${STRIMZI_NAMESPACE}"' in install
    assert 'watchNamespaces="{${APP_NAMESPACE}}"' in install


def test_every_release_is_installed_idempotently():
    script = (ROOT / "scripts" / "k8s_up.sh").read_text()
    commands = [line.strip() for line in script.splitlines() if line.strip().startswith("helm ")]
    installs = [c for c in commands if " install " in c or c.startswith("helm install")]
    assert installs == []
    upgrades = [c for c in commands if c.startswith("helm upgrade --install")]
    assert len(upgrades) == 4
    creates = [line for line in script.splitlines() if "kubectl create namespace" in line]
    assert creates and all("--dry-run=client" in line for line in creates)


def test_nightly_workflow_stays_dispatchable():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "k8s-nightly.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert "workflow_dispatch" in triggers and "push" not in triggers
    assert "pull_request" not in triggers
