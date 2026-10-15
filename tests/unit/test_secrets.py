"""Tests for secret validation, the env generator, and the delivery tracker."""

import pytest
from fastapi.testclient import TestClient

from driftguard.common.kafka import DeliveryError, DeliveryTracker
from driftguard.common.secrets import require_secret, secret_problem
from driftguard.common.settings import Settings
from driftguard.serving.app import create_app
from driftguard.streamfeatures.keys import HostKeyer

from .fakes import FakeRegistry, model_artifacts

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", ["", "   ", "change-me", "CHANGE-ME", "changeme", "short-one"])
def test_unusable_secrets_rejected(value):
    assert secret_problem(value) is not None
    with pytest.raises(ValueError, match="make env"):
        require_secret("X", value)


def test_strong_secret_accepted():
    assert secret_problem("a-long-random-secret-value") is None


def test_host_keyer_rejects_placeholder_salt():
    for salt in ("", "change-me", "short"):
        with pytest.raises(ValueError):
            HostKeyer(salt)
    assert len(HostKeyer("a-long-random-secret-value").key("10.0.0.1")) == 64


def test_admin_api_disabled_for_placeholder_token(tmp_path, repo_root):
    import fakeredis

    from driftguard.common.mlflow_client import CHAMPION, model_name

    registry = FakeRegistry()
    version = registry.register("cicids2017", model_artifacts(["a", "b"], 1), {"threshold": 0.5})
    registry.set_alias(model_name("cicids2017"), CHAMPION, version)
    settings = Settings(
        _env_file=None,
        ADMIN_TOKEN="change-me",
        DATASET="cicids2017",
        CONFIG_DIR=repo_root / "configs",
        PROMOTIONS_LOG_PATH=tmp_path / "p.jsonl",
    )
    app = create_app(
        settings, registry, fakeredis.FakeRedis(decode_responses=True), o_start_watcher=False
    )
    with TestClient(app) as client:
        response = client.post(
            "/admin/rollback",
            json={"dataset": "cicids2017"},
            headers={"Authorization": "Bearer change-me"},
        )
    assert response.status_code == 503 and "placeholder" in response.json()["detail"]


def test_init_env_generates_strong_secrets_and_keeps_other_lines():
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "init_env.py"
    spec = spec_from_file_location("init_env_script", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    example = (
        "DATASET=cicids2017\nADMIN_TOKEN=\nIP_HASH_SALT=change-me\nGF_ADMIN_PASSWORD="
        "already-a-strong-password-1\n"
    )
    rendered = module.render_env(example)
    lines = dict(line.split("=", 1) for line in rendered.splitlines())
    assert lines["DATASET"] == "cicids2017"
    assert (
        secret_problem(lines["ADMIN_TOKEN"]) is None
        and secret_problem(lines["IP_HASH_SALT"]) is None
    )
    assert lines["GF_ADMIN_PASSWORD"] == "already-a-strong-password-1"
    assert lines["ADMIN_TOKEN"] != lines["IP_HASH_SALT"]


def test_env_example_ships_no_usable_secret():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[2] / ".env.example").read_text()
    for key in ("ADMIN_TOKEN", "IP_HASH_SALT", "GF_ADMIN_PASSWORD"):
        line = next(item for item in text.splitlines() if item.startswith(f"{key}="))
        assert secret_problem(line.split("=", 1)[1]) is not None


class Msg:
    def topic(self):
        return "t"

    def partition(self):
        return 0


def test_delivery_tracker_raises_once_and_clears():
    tracker = DeliveryTracker()
    tracker(None, Msg())
    tracker.raise_if_failed()
    tracker("broker down", Msg())
    with pytest.raises(DeliveryError, match="failed delivery"):
        tracker.raise_if_failed()
    tracker.raise_if_failed()
