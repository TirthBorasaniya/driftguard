"""Tests for FastAPI serving layer endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.serving.main import app
    return TestClient(app)


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_has_required_fields(client):
    data = client.get("/health").json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert "uptime_seconds" in data


def test_predict_missing_fields_returns_422(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_valid_returns_200_or_503(client, sample_transaction_request):
    resp = client.post("/predict", json=sample_transaction_request)
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "threshold" in data
        assert 0 <= data["fraud_probability"] <= 1


def test_predict_explain_valid(client, sample_transaction_request):
    resp = client.post("/predict/explain", json=sample_transaction_request)
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


def test_predict_negative_amt_returns_422(client, sample_transaction_request):
    payload = {**sample_transaction_request, "amt": -10.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
