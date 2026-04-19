import pytest
from unittest.mock import patch, MagicMock
import asyncio
import numpy as np
from fastapi import Request, HTTPException
from fastapi.responses import Response
from fastapi.security import HTTPBasicCredentials

from app.main import (
    app,
    root,
    health_check,
    metrics,
    predict,
    predict_batch,
    admin_dashboard,
    get_admin_user,
)
from app.model import ChurnModel, get_model
from app.metrics import count_implemented_metrics, get_all_metrics
from app.middleware import MetricsMiddleware
from app.schemas import PredictionRequest, BatchPredictionRequest


def test_admin_auth_fail():
    with pytest.raises(HTTPException):
        get_admin_user(HTTPBasicCredentials(username="wrong", password="wrong"))


def test_metrics_disabled_branch():
    async def run():
        with patch("app.main.METRICS_ENABLED", False):
            with pytest.raises(HTTPException):
                await metrics()

    asyncio.run(run())


def test_model_not_loaded_branch():
    async def run():
        with patch("app.model.get_model") as mock_get:
            mock_model = mock_get.return_value
            # Use a mock that responds to bool() if needed
            mock_model.is_loaded = MagicMock(return_value=False)
            mock_model.is_loaded.__bool__.return_value = False
            with pytest.raises(HTTPException):
                await predict(MagicMock())

    asyncio.run(run())


def test_model_reload_fail_branches():
    model = ChurnModel()
    with patch("os.environ.get", return_value=None):
        model.reload("any")
        assert "DUMMY" in model.loaded_version
    with patch("mlflow.set_tracking_uri", side_effect=RuntimeError("MLflow Down")):
        model.reload("any")
        assert "DUMMY" in model.loaded_version


def test_guardrail_branches():
    async def run():
        # Build a valid schema-compliant dict for partial testing
        # Tenure fail
        req = MagicMock()
        req.model_dump.return_value = {
            "gender": "Male",
            "seniorcitizen": 0,
            "partner": "No",
            "dependents": "No",
            "tenure": -1,
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
            "totalcharges": 100.0,
        }
        with pytest.raises(HTTPException):
            await predict(req)
        # Total charges fail
        req.model_dump.return_value["tenure"] = 10
        req.model_dump.return_value["totalcharges"] = -50
        with pytest.raises(HTTPException):
            await predict(req)

    asyncio.run(run())


def test_batch_exception_branch():
    async def run():
        with patch(
            "app.model.ChurnModel.predict_with_latency", side_effect=Exception("Fail")
        ):
            # VERIFIED schema-compliant values
            req = PredictionRequest(
                gender="Male",
                seniorcitizen=0,
                partner="No",
                dependents="No",
                tenure=10,
                phoneservice="Yes",
                multiplelines="No",
                internetservice="DSL",
                onlinesecurity="No",
                onlinebackup="No",
                deviceprotection="No",
                techsupport="No",
                streamingtv="No",
                streamingmovies="No",
                contract="Month-to-month",
                paperlessbilling="Yes",
                paymentmethod="Electronic check",
                monthlycharges=50.0,
                totalcharges=500.0,
            )
            with pytest.raises(HTTPException):
                await predict_batch(BatchPredictionRequest(predictions=[req]))

    asyncio.run(run())


def test_shap_logic_deep():
    model = get_model()
    with patch("shap.Explainer") as mock_exp:
        mock_e = mock_exp.return_value
        mock_v = MagicMock()
        mock_v.values = np.array([[0.5] * 19])
        mock_e.return_value = mock_v

        if hasattr(model.model, "named_steps") and "pre" in model.model.named_steps:
            model.model.named_steps["pre"].get_feature_names_out = MagicMock(
                return_value=["f"] * 19
            )

        # Valid strings for raw dict bypass if model uses them
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


def test_middleware_errors():
    mock_app = MagicMock()
    middleware = MetricsMiddleware(mock_app)
    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}

    async def call_500(r):
        return Response(status_code=500)

    async def run():
        await middleware.dispatch(Request(scope=scope), call_500)

    asyncio.run(run())


def test_sync_helpers():
    count_implemented_metrics()
    get_all_metrics()

    async def run():
        await root()
        await health_check()
        await admin_dashboard("admin")
        with patch("app.main.METRICS_ENABLED", True):
            await metrics()

        req = PredictionRequest(
            gender="Male",
            seniorcitizen=0,
            partner="No",
            dependents="No",
            tenure=10,
            phoneservice="Yes",
            multiplelines="No",
            internetservice="DSL",
            onlinesecurity="No",
            onlinebackup="No",
            deviceprotection="No",
            techsupport="No",
            streamingtv="No",
            streamingmovies="No",
            contract="Month-to-month",
            paperlessbilling="Yes",
            paymentmethod="Electronic check",
            monthlycharges=50.0,
            totalcharges=500.0,
        )
        await predict(req)
        await predict_batch(BatchPredictionRequest(predictions=[req]))

    asyncio.run(run())
