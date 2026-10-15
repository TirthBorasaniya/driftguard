"""E3b: consumer autoscaling on Kafka lag in kind, with a duplicate and loss check (SPEC 9)."""

import json
import os
import subprocess
import threading
import time
from typing import Any

import yaml

from driftguard.common.logging import get_logger
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

logger = get_logger("benchmark.e3b")


# ============= Cluster helpers =============


def kubectl(*args: str, capture: bool = True) -> str:
    """Run kubectl against the repository kubeconfig."""
    env = {**os.environ, "KUBECONFIG": str(REPO_ROOT / ".kube" / "config")}
    result = subprocess.run(
        ["kubectl", *args], check=True, capture_output=capture, text=True, env=env
    )
    return result.stdout if capture else ""


def ready_replicas(namespace: str, deployment: str = "driftguard-consumer") -> int:
    """Ready replicas of a deployment."""
    out = kubectl(
        "get", "deployment", deployment, "-n", namespace, "-o", "jsonpath={.status.readyReplicas}"
    )
    return int(out or 0)


def job_manifest(config: dict, name: str, command: list[str], args: list[str]) -> str:
    """Render the chart's producer Job with a custom name, command, and arguments."""
    rendered = subprocess.run(
        [
            str(REPO_ROOT / ".cache" / "bin" / "helm")
            if (REPO_ROOT / ".cache/bin/helm").exists()
            else "helm",
            "template",
            "driftguard",
            str(REPO_ROOT / config["chart"]),
            "-f",
            str(REPO_ROOT / config["values"]),
            "--show-only",
            "templates/jobs.yaml",
            "--set",
            "producer.enabled=true",
            "--set",
            "bootstrap.enabled=false",
            "--set",
            "secrets.adminToken=unused",
            "--set",
            "secrets.ipHashSalt=unused",
            "--namespace",
            config["namespace"],
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    job = yaml.safe_load(rendered)
    job["metadata"]["name"] = name
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["command"], container["args"] = command, args
    return yaml.safe_dump(job)


def run_job(config: dict, name: str, command: list[str], args: list[str], timeout_s: float) -> str:
    """Create a Job, wait for completion, and return its logs."""
    manifest = job_manifest(config, name, command, args)
    env = {**os.environ, "KUBECONFIG": str(REPO_ROOT / ".kube" / "config")}
    subprocess.run(
        ["kubectl", "apply", "-n", config["namespace"], "-f", "-"],
        input=manifest,
        text=True,
        check=True,
        env=env,
        capture_output=True,
    )
    kubectl(
        "wait",
        f"job/{name}",
        "-n",
        config["namespace"],
        "--for=condition=complete",
        f"--timeout={int(timeout_s)}s",
    )
    return kubectl("logs", f"job/{name}", "-n", config["namespace"])


def prom_lag(prometheus_url: str, dataset: str) -> float:
    """Total consumer lag from Prometheus (0 when no series)."""
    import httpx

    response = httpx.get(
        f"{prometheus_url}/api/v1/query",
        timeout=10,
        params={"query": f'sum(driftguard_consumer_lag{{dataset="{dataset}"}})'},
    )
    response.raise_for_status()
    result = response.json()["data"]["result"]
    return float(result[0]["value"][1]) if result else 0.0


# ============= Analysis =============


def analyze(samples: list[dict], phases: list[dict], threshold: float) -> dict[str, Any]:
    """
    Derive scaling timings from samples of `{t_s, replicas, lag}`.

    Phase boundaries are taken from the configured durations relative to the run start.
    """
    step_up = phases[0]["duration_s"]
    step_down = step_up + phases[1]["duration_s"]
    before = max((s["replicas"] for s in samples if s["t_s"] < step_up), default=1)
    first_new = next(
        (s["t_s"] for s in samples if s["t_s"] >= step_up and s["replicas"] > before), None
    )
    high = [s for s in samples if s["t_s"] >= step_up]
    peak = max(high, key=lambda s: s["lag"], default=None)
    drained = next(
        (s["t_s"] for s in samples if s["t_s"] >= step_down and s["lag"] < threshold), None
    )
    after = [s for s in samples if s["t_s"] >= step_down]
    scaled_in = next((s["t_s"] for s in after if s["replicas"] <= before), None)
    return {
        "replicas_before_step": before,
        "max_replicas": max((s["replicas"] for s in samples), default=0),
        "time_to_first_new_pod_ready_s": None if first_new is None else first_new - step_up,
        "peak_lag": peak["lag"] if peak else None,
        "peak_lag_at_s": peak["t_s"] if peak else None,
        "time_to_drain_below_threshold_s": None if drained is None else drained - step_down,
        "scale_in_after_load_drop_s": None if scaled_in is None else scaled_in - step_down,
    }


# ============= Experiment =============


def run_e3b(config: dict, o_reduced: bool = False) -> dict[str, Any]:
    """Drive the phases with producer Jobs while sampling replicas and lag."""
    factor = float(config["reduced_duration_factor"]) if o_reduced else 1.0
    phases = [
        {"rate_eps": p["rate_eps"], "duration_s": p["duration_s"] * factor}
        for p in config["phases"]
    ]
    namespace, dataset = config["namespace"], config["dataset"]
    samples: list[dict] = []
    stop = threading.Event()
    started = time.monotonic()

    def sample() -> None:
        while not stop.is_set():
            samples.append(
                {
                    "t_s": round(time.monotonic() - started, 1),
                    "replicas": ready_replicas(namespace),
                    "lag": prom_lag(config["prometheus_url"], dataset),
                }
            )
            stop.wait(float(config["sample_interval_s"]))

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    slices, offset, run_tag = [], 0, int(time.time())
    for i, phase in enumerate(phases):
        count = int(phase["rate_eps"] * phase["duration_s"])
        slices.append((offset, count))
        logger.info("e3b_phase", phase=i, rate=phase["rate_eps"], duration=phase["duration_s"])
        run_job(
            config,
            f"e3b-producer-{run_tag}-{i}",
            ["python", "-m", "driftguard.producer.replay"],
            [
                "--mode",
                "fixed_rate",
                "--rate",
                str(phase["rate_eps"]),
                "--start-index",
                str(offset),
                "--limit",
                str(count),
                "--duration-s",
                str(phase["duration_s"]),
            ],
            timeout_s=phase["duration_s"] + 600,
        )
        offset += count
    time.sleep(float(config["post_observe_s"]) * factor)
    stop.set()
    sampler.join(timeout=30)
    logs = run_job(
        config,
        f"e3b-check-{run_tag}",
        ["python", "-m", "driftguard.benchmark.k8s_check"],
        ["--dataset", dataset, "--slices", ",".join(f"{s}:{c}" for s, c in slices)],
        timeout_s=3600,
    )
    check = next(
        json.loads(line)["e3b_check"]
        for line in logs.splitlines()
        if line.startswith('{"e3b_check"')
    )
    return {
        "phases": phases,
        "reduced": o_reduced,
        "scaling": analyze(samples, phases, float(config["lag_threshold"])),
        "duplicate_and_loss_check": check,
        "samples": samples,
        "keda": {
            "lagThreshold": 5000,
            "minReplicaCount": 1,
            "maxReplicaCount": 6,
            "cooldownPeriod": 60,
        },
    }
