"""Tests for the FastAPI serving layer."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the API."""
    from src.serving.api import app

    return TestClient(app)


def test_health_endpoint(client):
    """Health endpoint must return 200 with required fields."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_version" in data
    assert "uptime_seconds" in data
    assert "total_predictions" in data


def test_health_status_value(client):
    """Health status must be 'healthy'."""
    response = client.get("/health")
    assert response.json()["status"] == "healthy"


def test_predict_missing_required_fields(client):
    """Predict with missing required fields must return 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_missing_transaction_dt(client):
    """Predict with only TransactionAmt must return 422."""
    response = client.post("/predict", json={"TransactionAmt": 100.0})
    assert response.status_code == 422


def test_predict_valid_input(client):
    """Predict with valid input must return 200 or 503 (if model not loaded)."""
    payload = {
        "TransactionAmt": 150.0,
        "TransactionDT": 86400.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "transaction_id" in data
        assert "model_version" in data
        assert 0 <= data["fraud_probability"] <= 1


def test_predict_with_extra_features(client):
    """Predict with additional features must not cause errors."""
    payload = {
        "TransactionAmt": 150.0,
        "TransactionDT": 86400.0,
        "ProductCD": "W",
        "card4": "visa",
        "C1": 5.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]


def test_recent_predictions(client):
    """Recent predictions endpoint must return a list."""
    response = client.get("/predictions/recent")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_prediction_stats(client):
    """Prediction stats endpoint must return a dict."""
    response = client.get("/predictions/stats")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_drift_latest(client):
    """Drift latest endpoint must return a dict."""
    response = client.get("/drift/latest")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_healing_events(client):
    """Healing events endpoint must return a list."""
    response = client.get("/healing/events")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
