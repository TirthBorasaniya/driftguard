"""Tests for the FastAPI serving layer (skipped when serving dependencies are unavailable)."""

import pytest


@pytest.fixture
def client():
    # serving imports the full inference stack; skip cleanly if any dep is absent
    pytest.importorskip("fastapi")
    pytest.importorskip("mlflow")
    pytest.importorskip("aiosqlite")
    pytest.importorskip("prometheus_fastapi_instrumentator")
    from fastapi.testclient import TestClient

    from src.serving.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_has_required_fields(client):
    data = client.get("/health").json()
    assert {"status", "model_loaded", "model_version", "uptime_seconds"} <= set(data)


def test_predict_missing_fields_returns_422(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_valid_returns_200_or_503(client, sample_flow_request):
    resp = client.post("/predict", json=sample_flow_request)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert {"event_id", "anomaly_score", "is_anomaly", "threshold"} <= set(data)
        assert 0.0 <= data["anomaly_score"] <= 1.0
        assert isinstance(data["is_anomaly"], bool)


def test_predict_explain_valid(client, sample_flow_request):
    resp = client.post("/predict/explain", json=sample_flow_request)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "top_features" in data
        assert isinstance(data["top_features"], list)


def test_recent_predictions_returns_list(client):
    resp = client.get("/predictions/recent")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_prediction_stats_returns_dict(client):
    resp = client.get("/predictions/stats")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)
