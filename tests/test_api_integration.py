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
        "gender": "Female",
        "seniorcitizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 1,
        "phoneservice": "No",
        "multiplelines": "No phone service",
        "internetservice": "DSL",
        "onlinesecurity": "No",
        "onlinebackup": "Yes",
        "deviceprotection": "No",
        "techsupport": "No",
        "streamingtv": "No",
        "streamingmovies": "No",
        "contract": "Month-to-month",
        "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check",
        "monthlycharges": 29.85,
        "totalcharges": 29.85,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "is_churn" in data
    assert "risk_tier" in data
    assert "reason_codes" in data
    assert isinstance(data["reason_codes"], list)


def test_predict_invalid_input(client):
    """Test prediction with invalid tenure (Guardrail test)."""
    payload = {
        "gender": "Female",
        "seniorcitizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": -5,  # Invalid tenure
        "phoneservice": "No",
        "multiplelines": "No phone service",
        "internetservice": "DSL",
        "onlinesecurity": "No",
        "onlinebackup": "Yes",
        "deviceprotection": "No",
        "techsupport": "No",
        "streamingtv": "No",
        "streamingmovies": "No",
        "contract": "Month-to-month",
        "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check",
        "monthlycharges": 29.85,
        "totalcharges": 29.85,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_health_check(client):
    """Verify health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "unhealthy"]
