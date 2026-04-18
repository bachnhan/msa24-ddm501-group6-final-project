"""
Integration tests for the Churn Prediction API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_predict_endpoint_success(client):
    """Test standard prediction request."""
    payload = {
        "age": 30,
        "gender": "Female",
        "tenure": 24,
        "usage_frequency": 20,
        "support_calls": 1,
        "payment_delay": 0,
        "subscription_type": "Premium",
        "contract_length": "Annual",
        "total_spend": 1200.0,
        "last_interaction": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "is_churn" in data
    assert "latency_ms" in data

def test_predict_invalid_input(client):
    """Test prediction with invalid age (Guardrail test)."""
    payload = {
        "age": -10,  # Invalid age
        "gender": "Female",
        "tenure": 24,
        "usage_frequency": 20,
        "support_calls": 1,
        "payment_delay": 0,
        "subscription_type": "Premium",
        "contract_length": "Annual",
        "total_spend": 1200.0,
        "last_interaction": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Guardrail: Age" in response.json()["detail"]

def test_health_check(client):
    """Verify health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "unhealthy"]
