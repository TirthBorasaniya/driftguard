"""Static checks of the application image and Compose wiring (no container runtime needed)."""

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "docker" / "app.Dockerfile").read_text()
DEFAULT_UID = 10001


def test_service_user_takes_the_host_uid_and_gid_through_build_args():
    """Regression: the fixed uid 10001 could not write the host-owned bind mount ./results on a
    Linux runner (PermissionError on /app/results/promotions.jsonl)."""
    assert re.search(rf"^ARG APP_UID={DEFAULT_UID}$", DOCKERFILE, re.M)
    assert re.search(rf"^ARG APP_GID={DEFAULT_UID}$", DOCKERFILE, re.M)
    assert '--uid "${APP_UID}"' in DOCKERFILE and '--gid "${APP_GID}"' in DOCKERFILE
    assert "--uid 10001" not in DOCKERFILE
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    args = compose["x-app"]["build"]["args"]
    assert args == {"APP_UID": "${HOST_UID:-10001}", "APP_GID": "${HOST_GID:-10001}"}


def test_no_permission_is_widened():
    for path in (
        ROOT / "docker" / "app.Dockerfile",
        ROOT / "docker-compose.yml",
        ROOT / "Makefile",
        ROOT / ".github" / "workflows" / "integration.yml",
    ):
        text = path.read_text()
        assert not re.search(r"chmod\s+(-R\s+)?(0?777|0?666|a\+[rwx]*w|o\+[rwx]*w)", text), path
    assert "privileged" not in (ROOT / "docker-compose.yml").read_text()


def test_root_uid_is_refused_at_build_time():
    assert "must not be 0" in DOCKERFILE


def test_kubernetes_chart_runs_as_the_image_default_uid():
    helpers = (
        ROOT / "deploy" / "k8s" / "charts" / "driftguard" / "templates" / "_helpers.tpl"
    ).read_text()
    assert f"runAsUser: {DEFAULT_UID}" in helpers and f"fsGroup: {DEFAULT_UID}" in helpers


def test_git_python_refresh_is_quiet_in_the_image():
    assert re.search(r"^ENV GIT_PYTHON_REFRESH=quiet", DOCKERFILE, re.M)


def test_integration_workflow_exports_the_runner_ids_and_stays_manual():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "integration.yml").read_text())
    steps = workflow["jobs"]["compose"]["steps"]
    names = [step.get("name", "") for step in steps]
    export = next(s for s in steps if s.get("name") == "Run the containers as the runner user")
    assert "HOST_UID=$(id -u)" in export["run"] and "HOST_GID=$(id -g)" in export["run"]
    assert "mkdir -p data/raw data/processed results" in export["run"]
    assert names.index(export["name"]) < names.index("Start the stack")
    assert set(workflow.get("on", workflow.get(True))) == {"workflow_dispatch"}


def test_make_exports_the_current_user_ids():
    makefile = (ROOT / "Makefile").read_text()
    assert "export HOST_UID ?= $(shell id -u)" in makefile
    assert "export HOST_GID ?= $(shell id -g)" in makefile
