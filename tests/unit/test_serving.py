"""Tests for the serving router: scoring, shadow, canary split, admin token, rollback, watcher."""

import fakeredis
import numpy as np
import pytest
from fastapi.testclient import TestClient

from driftguard.common.mlflow_client import CHALLENGER, CHAMPION, PREVIOUS, model_name
from driftguard.common.promotions import read_promotions
from driftguard.common.settings import Settings
from driftguard.serving.app import create_app
from driftguard.serving.router import canary_bucket, routes_to_canary

from .fakes import FakeRegistry, model_artifacts

pytestmark = pytest.mark.unit
FEATURES = ["a", "b", "c"]
NAME = model_name("cicids2017")
TOKEN = "test-admin-token"


@pytest.fixture
def env(tmp_path, repo_root):
    registry = FakeRegistry()
    v1 = registry.register("cicids2017", model_artifacts(FEATURES, 1), {"threshold": 0.5})
    registry.set_alias(NAME, CHAMPION, v1)
    redis = fakeredis.FakeRedis(decode_responses=True)
    settings = Settings(
        _env_file=None,
        ADMIN_TOKEN=TOKEN,
        DATASET="cicids2017",
        CONFIG_DIR=repo_root / "configs",
        PROMOTIONS_LOG_PATH=tmp_path / "promotions.jsonl",
    )
    app = create_app(settings, registry, redis, o_start_watcher=False)
    with TestClient(app) as client:
        yield client, registry, redis, settings, app


def body(n=4, names=None, extra_feature=False):
    names = list(names or FEATURES) + (["zz"] if extra_feature else [])
    rng = np.random.default_rng(0)
    return {
        "dataset": "cicids2017",
        "feature_names": names,
        "events": [
            {"event_id": f"e{i}", "features": rng.normal(size=len(names)).tolist()}
            for i in range(n)
        ],
    }


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def test_score_returns_champion_results_and_selects_columns_by_name(env):
    client, *_ = env
    response = client.post("/score", json=body(extra_feature=True))
    assert response.status_code == 200
    data = response.json()
    assert data["rollout_mode"] == "champion_only" and data["latency_ms"] >= 0
    assert {r["model_alias"] for r in data["results"]} == {"champion"}
    assert all(r["decision"] == int(r["score"] >= 0.5) for r in data["results"])
    reordered = body(names=["c", "a", "b"])
    assert client.post("/score", json=reordered).status_code == 200


def test_request_validation(env):
    client, *_ = env
    assert client.post("/score", json=body(n=257)).status_code == 422
    bad = body()
    bad["events"][0]["features"] = [1.0]
    assert client.post("/score", json=bad).status_code == 422
    missing = body(names=["a", "b", "x"])
    assert client.post("/score", json=missing).status_code == 422
    assert client.post("/score", json={**body(), "dataset": "kyoto"}).status_code == 400


def test_health_ready_metrics_recent(env):
    client, *_ = env
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    client.post("/score", json=body())
    assert "driftguard_scoring_latency_seconds" in client.get("/metrics").text
    recent = client.get("/predictions/recent?limit=2").json()["predictions"]
    assert len(recent) == 2
    assert set(recent[0]) == {"event_id", "score", "decision", "model_alias", "model_version"}


def test_admin_requires_token(env):
    client, *_ = env
    payload = {"dataset": "cicids2017", "mode": "shadow"}
    assert client.post("/admin/rollout", json=payload).status_code == 401
    assert (
        client.post(
            "/admin/rollout", json=payload, headers={"Authorization": "Bearer nope"}
        ).status_code
        == 401
    )
    assert client.post("/admin/rollout", json=payload, headers=auth()).status_code == 200
    bad = {"dataset": "cicids2017", "mode": "yolo"}
    assert client.post("/admin/rollout", json=bad, headers=auth()).status_code == 422


def test_shadow_and_canary(env):
    client, registry, redis, _, app = env
    assert client.post("/score/shadow", json=body()).status_code == 409
    v2 = registry.register("cicids2017", model_artifacts(FEATURES, 2), {"threshold": 0.4})
    registry.set_alias(NAME, CHALLENGER, v2)
    app.state.store.refresh()
    client.post("/admin/rollout", json={"dataset": "cicids2017", "mode": "shadow"}, headers=auth())
    shadow = client.post("/score/shadow", json=body())
    assert shadow.status_code == 200 and shadow.json()["model_version"] == v2
    assert {r["model_alias"] for r in client.post("/score", json=body()).json()["results"]} == {
        "champion"
    }
    client.post(
        "/admin/rollout",
        json={"dataset": "cicids2017", "mode": "canary", "canary_fraction": 0.5},
        headers=auth(),
    )
    data = client.post("/score", json=body(n=200)).json()
    for result in data["results"]:
        expected = "challenger" if routes_to_canary(result["event_id"], 0.5) else "champion"
        assert result["model_alias"] == expected
        assert result["threshold"] == (0.4 if expected == "challenger" else 0.5)


def test_canary_hash_is_deterministic_and_uniform():
    ids = [f"event-{i}" for i in range(20000)]
    share = np.mean([routes_to_canary(e, 0.1) for e in ids])
    assert abs(share - 0.1) < 0.01
    assert canary_bucket("x") == canary_bucket("x")


def test_rollback_restores_previous_and_logs(env):
    client, registry, redis, settings, app = env
    assert (
        client.post("/admin/rollback", json={"dataset": "cicids2017"}, headers=auth()).status_code
        == 409
    )
    v2 = registry.register("cicids2017", model_artifacts(FEATURES, 3), {"threshold": 0.6})
    registry.set_alias(NAME, PREVIOUS, "1")
    registry.set_alias(NAME, CHAMPION, v2)
    app.state.store.refresh()
    assert client.post("/score", json=body()).json()["results"][0]["model_version"] == v2
    response = client.post("/admin/rollback", json={"dataset": "cicids2017"}, headers=auth())
    assert response.status_code == 200 and response.json()["champion"] == "1"
    app.state.store.refresh()
    assert client.post("/score", json=body()).json()["results"][0]["model_version"] == "1"
    assert registry.get_alias(NAME, PREVIOUS) is None
    assert redis.get("dg:rollout:cicids2017") == "champion_only"
    log = read_promotions(settings.promotions_log_path)
    assert log[-1]["action"] == "rollback" and log[-1]["rolled_back_from"] == v2


def test_inflight_model_reference_survives_swap(env):
    client, registry, _, _, app = env
    old = app.state.store.get(CHAMPION)
    v2 = registry.register("cicids2017", model_artifacts(FEATURES, 4), {"threshold": 0.5})
    registry.set_alias(NAME, CHAMPION, v2)
    assert app.state.store.refresh() == ["champion"]
    assert app.state.store.get(CHAMPION).version == v2
    x = np.zeros((3, 3), dtype=np.float32)
    assert old.score(tuple(FEATURES), x).shape == (3,)
