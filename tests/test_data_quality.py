"""
Data Quality Tests for the Customer Churn Prediction System.
Verifies input data integrity and model output constraints.
"""

import pytest
from app.schemas import PredictionRequest
from app.model import get_model
from pathlib import Path

def test_data_schema_validation():
    """Verify that Pydantic schemas catch invalid inputs."""
    # Invalid tenure type (should be int)
    with pytest.raises(Exception):
        PredictionRequest(
            gender="Female", seniorcitizen=0, partner="Yes", dependents="No",
            tenure="one", # Should be int
            phoneservice="No", multiplelines="No phone service",
            internetservice="DSL", onlinesecurity="No", onlinebackup="Yes",
            deviceprotection="No", techsupport="No", streamingtv="No",
            streamingmovies="No", contract="Month-to-month", paperlessbilling="Yes",
            paymentmethod="Electronic check", monthlycharges=29.85, totalcharges=29.85
        )
    
    # Missing required field (contract)
    with pytest.raises(Exception):
        PredictionRequest(
            gender="Female", seniorcitizen=0, partner="Yes", dependents="No",
            tenure=1, phoneservice="No", multiplelines="No phone service",
            internetservice="DSL", onlinesecurity="No", onlinebackup="Yes",
            deviceprotection="No", techsupport="No", streamingtv="No",
            streamingmovies="No", # contract missing
            paperlessbilling="Yes", paymentmethod="Electronic check",
            monthlycharges=29.85, totalcharges=29.85
        )

def test_model_output_range():
    """Verify that model churn probability is within [0, 1]."""
    model = get_model()
    if not model.is_loaded():
        pytest.skip("Model not loaded. CI environment might use dummy fallback.")
        
    sample_data = {
        "gender": "Female", "seniorcitizen": 0, "partner": "Yes", "dependents": "No",
        "tenure": 1, "phoneservice": "No", "multiplelines": "No phone service",
        "internetservice": "DSL", "onlinesecurity": "No", "onlinebackup": "Yes",
        "deviceprotection": "No", "techsupport": "No", "streamingtv": "No",
        "streamingmovies": "No", "contract": "Month-to-month", "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check", "monthlycharges": 29.85, "totalcharges": 29.85
    }
    is_churn, prob, risk_tier, reason_codes, latency = model.predict_with_latency(sample_data)
    
    assert 0.0 <= prob <= 1.0
    assert isinstance(is_churn, bool)
    assert risk_tier in ["High", "Medium", "Low"]
    assert isinstance(reason_codes, list)

def test_data_completeness():
    """Verify essential project files exist."""
    assert Path("requirements.txt").exists()
    assert Path("app/main.py").exists()
    assert Path("docker-compose.yml").exists()
