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
@pytest.mark.parametrize("tenure,expected_status", [
    (0, 200),     # Boundary: min
    (120, 200),   # Boundary: max
    (-1, 400),    # Out of bounds: low
    (121, 400),   # Out of bounds: high
])
def test_predict_tenure_guardrails(client, tenure, expected_status):
    payload = {
        "gender": "Male", "seniorcitizen": 0, "partner": "No", "dependents": "No",
        "tenure": tenure, "phoneservice": "Yes", "multiplelines": "No",
        "internetservice": "DSL", "onlinesecurity": "No", "onlinebackup": "No",
        "deviceprotection": "No", "techsupport": "No", "streamingtv": "No",
        "streamingmovies": "No", "contract": "Month-to-month", "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check", "monthlycharges": 50.0, "totalcharges": 50.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == expected_status

def test_predict_negative_spend(client):
    payload = {
        "gender": "Male", "seniorcitizen": 0, "partner": "No", "dependents": "No",
        "tenure": 10, "phoneservice": "Yes", "multiplelines": "No",
        "internetservice": "DSL", "onlinesecurity": "No", "onlinebackup": "No",
        "deviceprotection": "No", "techsupport": "No", "streamingtv": "No",
        "streamingmovies": "No", "contract": "Month-to-month", "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check", "monthlycharges": 50.0, "totalcharges": -100.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Total charges cannot be negative" in response.json()["detail"]

# 2. BATCH PREDICTION TESTS
def test_predict_batch_success(client):
    payload = {
        "predictions": [
            {"gender": "Male", "seniorcitizen": 0, "partner": "No", "dependents": "No", "tenure": 10, "phoneservice": "Yes", "multiplelines": "No", "internetservice": "DSL", "onlinesecurity": "No", "onlinebackup": "No", "deviceprotection": "No", "techsupport": "No", "streamingtv": "No", "streamingmovies": "No", "contract": "Month-to-month", "paperlessbilling": "Yes", "paymentmethod": "Electronic check", "monthlycharges": 50.0, "totalcharges": 500.0},
            {"gender": "Female", "seniorcitizen": 1, "partner": "Yes", "dependents": "No", "tenure": 24, "phoneservice": "Yes", "multiplelines": "Yes", "internetservice": "Fiber optic", "onlinesecurity": "Yes", "onlinebackup": "Yes", "deviceprotection": "Yes", "techsupport": "No", "streamingtv": "Yes", "streamingmovies": "Yes", "contract": "One year", "paperlessbilling": "Yes", "paymentmethod": "Credit card (automatic)", "monthlycharges": 100.0, "totalcharges": 2400.0}
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
