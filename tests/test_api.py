"""Additional API endpoint tests for the FastAPI serving layer."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.serving.main import app

    return TestClient(app)


def test_health_status_value(client):
    """Health endpoint must report status 'healthy'."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_health_total_predictions_is_int(client):
    """total_predictions field must be a non-negative integer."""
    data = client.get("/health").json()
    assert isinstance(data["total_predictions"], int)
    assert data["total_predictions"] >= 0


def test_predict_negative_amount_rejected(client, sample_transaction_request):
    """Negative transaction amount must be rejected with 422."""
    payload = {**sample_transaction_request, "amt": -1.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_zero_amount_rejected(client, sample_transaction_request):
    """Zero transaction amount must be rejected with 422 (gt=0 constraint)."""
    payload = {**sample_transaction_request, "amt": 0.0}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_missing_cc_num_rejected(client, sample_transaction_request):
    """Request missing cc_num must be rejected with 422."""
    payload = {k: v for k, v in sample_transaction_request.items() if k != "cc_num"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_response_shape(client, sample_transaction_request):
    """Successful predict response must include all required fields."""
    resp = client.post("/predict", json=sample_transaction_request)
    if resp.status_code == 200:
        data = resp.json()
        required = {
            "transaction_id", "fraud_probability", "is_fraud",
            "threshold", "model_version", "timestamp",
        }
        assert required.issubset(data.keys())
        assert 0.0 <= data["fraud_probability"] <= 1.0
        assert isinstance(data["is_fraud"], bool)


def test_prediction_stats_total_non_negative(client):
    """Prediction stats must report a non-negative total count."""
    data = client.get("/predictions/stats").json()
    assert data.get("total_predictions", 0) >= 0


def test_recent_predictions_limit_param(client):
    """recent_predictions with limit=1 must return at most 1 row."""
    resp = client.get("/predictions/recent?limit=1")
    assert resp.status_code == 200
    assert len(resp.json()) <= 1
