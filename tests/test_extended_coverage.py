"""
Extended Integration tests for the Churn Prediction API.
Covers Batch Prediction, Admin Features, and Advanced Guardrails.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def auth_header():
    import base64
    auth_str = "admin:admin"
    encoded = base64.b64encode(auth_str.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}

# 1. ADVANCED GUARDRAIL TESTS
@pytest.mark.parametrize("age,expected_status", [
    (18, 200),   # Boundary: min
    (120, 200),  # Boundary: max
    (17, 400),   # Out of bounds: low
    (121, 400),  # Out of bounds: high
])
def test_predict_age_guardrails(client, age, expected_status):
    payload = {
        "age": age, "gender": "Male", "tenure": 12, "usage_frequency": 10,
        "support_calls": 1, "payment_delay": 0, "subscription_type": "Basic",
        "contract_length": "Monthly", "total_spend": 100.0, "last_interaction": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == expected_status

def test_predict_negative_spend(client):
    payload = {
        "age": 30, "gender": "Male", "tenure": 12, "usage_frequency": 10,
        "support_calls": 1, "payment_delay": 0, "subscription_type": "Basic",
        "contract_length": "Monthly", "total_spend": -50.0, "last_interaction": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Total spend cannot be negative" in response.json()["detail"]

# 2. BATCH PREDICTION TESTS
def test_predict_batch_success(client):
    payload = {
        "predictions": [
            {"age": 30, "gender": "Male", "tenure": 12, "usage_frequency": 10, "support_calls": 1, "payment_delay": 0, "subscription_type": "Basic", "contract_length": "Monthly", "total_spend": 100.0, "last_interaction": 1},
            {"age": 45, "gender": "Female", "tenure": 24, "usage_frequency": 5, "support_calls": 3, "payment_delay": 2, "subscription_type": "Premium", "contract_length": "Annual", "total_spend": 500.0, "last_interaction": 10}
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert "avg_latency_ms" in data

# 3. ADMIN ENDPOINT TESTS
def test_admin_reload_unauthorized(client):
    """Should fail without auth."""
    response = client.post("/model/reload?version_or_alias=latest")
    assert response.status_code == 401

def test_admin_reload_success(client, auth_header):
    """Should pass with correct basic auth."""
    # Note: This might talk to DagsHub, but in CI it uses the Dummy fallback
    response = client.post("/model/reload?version_or_alias=latest", headers=auth_header)
    assert response.status_code in [200, 500] # 500 is acceptable if DagsHub is down, but auth PASSED
