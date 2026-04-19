import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from app.model import ChurnModel
import os

client = TestClient(app)


@pytest.fixture
def auth_header():
    import base64

    auth_str = "admin:admin"
    encoded = base64.b64encode(auth_str.encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


# 1. Admin Dashboard HTML coverage
def test_admin_dashboard_html(auth_header):
    response = client.get("/admin", headers=auth_header)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# 2. Root endpoint coverage (Hits main.py 83-84)
def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Customer Churn Prediction System" in response.json()["name"]


# 3. Model not loaded coverage (Hits main.py 115)
def test_predict_model_not_loaded():
    with patch("app.model.ChurnModel.is_loaded", return_value=False):
        payload = {
            "gender": "Male",
            "seniorcitizen": 0,
            "partner": "No",
            "dependents": "No",
            "tenure": 10,
            "phoneservice": "Yes",
            "multiplelines": "No",
            "internetservice": "DSL",
            "onlinesecurity": "No",
            "onlinebackup": "No",
            "deviceprotection": "No",
            "techsupport": "No",
            "streamingtv": "No",
            "streamingmovies": "No",
            "contract": "Month-to-month",
            "paperlessbilling": "Yes",
            "paymentmethod": "Electronic check",
            "monthlycharges": 50.0,
            "totalcharges": 500.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 503


# 4. Metrics disabled coverage (Hits main.py 107)
def test_metrics_disabled():
    with patch("app.main.METRICS_ENABLED", False):
        response = client.get("/metrics")
        assert response.status_code == 503


# 5. Dummy Model Fallback coverage
def test_dummy_model_initialization():
    with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": ""}):
        model = ChurnModel()
        assert "DUMMY" in model.loaded_version
        data = {
            "gender": "Male",
            "seniorcitizen": 0,
            "partner": "No",
            "dependents": "No",
            "tenure": 10,
            "phoneservice": "Yes",
            "multiplelines": "No",
            "internetservice": "DSL",
            "onlinesecurity": "No",
            "onlinebackup": "No",
            "deviceprotection": "No",
            "techsupport": "No",
            "streamingtv": "No",
            "streamingmovies": "No",
            "contract": "Month-to-month",
            "paperlessbilling": "Yes",
            "paymentmethod": "Electronic check",
            "monthlycharges": 50.0,
            "totalcharges": 500.0,
        }
        is_churn, prob, risk, reasons, latency = model.predict_with_latency(data)
        assert "analysis_unavailable" in reasons


# 6. SHAP Path Brute Force
def test_model_internal_shap_logic():
    from app.model import get_model

    model = get_model()
    with patch("shap.Explainer") as mock_explainer:
        mock_e = mock_explainer.return_value
        mock_v = MagicMock()
        # Mock 19 columns
        mock_v.values = [[0.1] * 19]
        mock_e.return_value = mock_v
        model.model.named_steps["pre"].get_feature_names_out = MagicMock(
            return_value=["col"] * 19
        )

        data = {
            "gender": "Male",
            "seniorcitizen": 0,
            "partner": "No",
            "dependents": "No",
            "tenure": 10,
            "phoneservice": "Yes",
            "multiplelines": "No",
            "internetservice": "DSL",
            "onlinesecurity": "No",
            "onlinebackup": "No",
            "deviceprotection": "No",
            "techsupport": "No",
            "streamingtv": "No",
            "streamingmovies": "No",
            "contract": "Month-to-month",
            "paperlessbilling": "Yes",
            "paymentmethod": "Electronic check",
            "monthlycharges": 50.0,
            "totalcharges": 500.0,
        }
        model.predict_with_latency(data)


# 7. Middleware Labels (Hits middleware.py loops)
def test_middleware_label_recording():
    # Record a few requests to trigger the label processing logic
    client.get("/health")
    client.post("/predict/batch", json={"predictions": []})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


# 8. Admin Auth failures
def test_admin_auth_wrong_creds():
    response = client.get(
        "/admin", headers={"Authorization": "Basic d3Jvbmc6d3Jvbmc="}
    )  # wrong:wrong
    assert response.status_code == 401
