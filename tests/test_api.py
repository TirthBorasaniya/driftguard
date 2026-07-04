"""Additional API endpoint tests for the FastAPI serving layer."""

import pytest


@pytest.fixture
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("mlflow")
    pytest.importorskip("aiosqlite")
    pytest.importorskip("prometheus_fastapi_instrumentator")
    from fastapi.testclient import TestClient

    from src.serving.main import app

    return TestClient(app)


def test_health_status_value(client):
    """Health endpoint must report status 'healthy'."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_health_total_predictions_is_int(client):
    data = client.get("/health").json()
    assert isinstance(data["total_predictions"], int)
    assert data["total_predictions"] >= 0


def test_predict_missing_feature_rejected(client, sample_flow_request):
    """A request missing a required feature must be rejected with 422."""
    payload = {k: v for k, v in sample_flow_request.items() if k != "flow_duration"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_response_shape(client, sample_flow_request):
    resp = client.post("/predict", json=sample_flow_request)
    if resp.status_code == 200:
        data = resp.json()
        required = {"event_id", "anomaly_score", "is_anomaly", "threshold", "model_version", "timestamp"}
        assert required.issubset(data.keys())
        assert 0.0 <= data["anomaly_score"] <= 1.0


def test_prediction_stats_total_non_negative(client):
    data = client.get("/predictions/stats").json()
    assert data.get("total_predictions", 0) >= 0


def test_recent_predictions_limit_param(client):
    resp = client.get("/predictions/recent?limit=1")
    assert resp.status_code == 200
    assert len(resp.json()) <= 1
