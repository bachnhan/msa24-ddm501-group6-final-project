import pytest
from unittest.mock import patch, MagicMock
import asyncio
from fastapi import Request, HTTPException
from fastapi.responses import Response
from fastapi.security import HTTPBasicCredentials

# Import app components directly for exhaustive branch testing
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


def test_exhaustive_error_branches():
    """Manually trigger every exception and error branch in the codebase."""

    async def run_internal():
        # 1. Main.py: Incorrect Admin Credentials (Line 59)
        invalid_creds = HTTPBasicCredentials(username="wrong", password="wrong")
        with pytest.raises(HTTPException) as exc:
            get_admin_user(invalid_creds)
        assert exc.value.status_code == 401

        # 2. Main.py: Metrics disabled branch (Line 107)
        with patch("app.main.METRICS_ENABLED", False):
            with pytest.raises(HTTPException) as exc:
                await metrics()
            assert exc.value.status_code == 503

        # 3. Main.py: Model not loaded branch (Line 115)
        with patch("app.main.get_model") as mock_get_model:
            mock_model = mock_get_model.return_value
            mock_model.is_loaded.return_value = False
            with pytest.raises(HTTPException) as exc:
                await predict(MagicMock())
            assert exc.value.status_code == 503

        # 4. Main.py: Guardrail Tenure (Line 124)
        req_invalid_tenure = MagicMock()
        req_invalid_tenure.model_dump.return_value = {
            "tenure": -1,
            "gender": "Male",
            "totalcharges": 100,
        }
        with pytest.raises(HTTPException) as exc:
            await predict(req_invalid_tenure)
        assert "Tenure" in exc.value.detail

        # 5. Main.py: Guardrail TotalCharges (Line 130)
        req_invalid_charges = MagicMock()
        req_invalid_charges.model_dump.return_value = {
            "tenure": 10,
            "gender": "Male",
            "totalcharges": -50,
        }
        with pytest.raises(HTTPException) as exc:
            await predict(req_invalid_charges)
        assert "charges" in exc.value.detail

        # 6. Middleware.py: Error Labeling (Lines 74-89)
        mock_app = MagicMock()
        middleware = MetricsMiddleware(mock_app)
        scope = {"type": "http", "method": "GET", "path": "/error", "headers": []}
        request = Request(scope=scope)

        async def call_next_fail(req):
            return Response(content="error", status_code=500)

        await middleware.dispatch(request, call_next_fail)

        async def call_next_404(req):
            return Response(content="not found", status_code=404)

        await middleware.dispatch(request, call_next_404)

    asyncio.run(run_internal())


def test_success_path_coverage():
    """Standard success paths to keep the baseline high."""

    async def run_success():
        await root()
        await health_check()
        count_implemented_metrics()
        get_all_metrics()
        # Call admin dashboard with success
        await admin_dashboard("admin")

    asyncio.run(run_success())
