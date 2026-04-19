"""
Unit Tests for Model Functions (as requested in Slide).
Tests preprocessor, predict, and load_model logic.
"""

import pytest
import pandas as pd
from app.model import ChurnModel, get_model


def test_model_initialization():
    """Test load_model() functionality (called in __init__)."""
    model = get_model()
    assert model is not None
    # Verify that model has loaded (either real or dummy)
    assert model.is_loaded() is True
    assert model.loaded_version != "initializing..."


def test_predict_logic_unit():
    """Test the predict() function of the ChurnModel."""
    model = get_model()

    sample_input = {
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

    is_churn, risk_tier, reason_codes = model.predict(sample_input)

    assert isinstance(is_churn, bool)
    assert risk_tier in ["High", "Medium", "Low"]
    assert isinstance(reason_codes, list)


def test_preprocessor_unit():
    """Test that the internal preprocessor handles numeric conversions."""
    # This specifically tests that predict_with_latency handles data correctly
    model = get_model()

    # Test data with different types that should be handled by the pipeline
    sample_input = {
        "gender": "Male",
        "seniorcitizen": 1,
        "partner": "No",
        "dependents": "No",
        "tenure": 12,
        "phoneservice": "Yes",
        "multiplelines": "No",
        "internetservice": "Fiber optic",
        "onlinesecurity": "No",
        "onlinebackup": "No",
        "deviceprotection": "No",
        "techsupport": "No",
        "streamingtv": "No",
        "streamingmovies": "No",
        "contract": "Month-to-month",
        "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check",
        "monthlycharges": 70.0,
        "totalcharges": 840.0,
    }

    # This ensures the pipeline's transform step is successful
    res = model.predict_with_latency(sample_input)
    assert len(res) == 5
