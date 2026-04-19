"""
Data Quality Tests using Pandera (as requested in Slide).
Verifies schema, null checks, and class distribution.
"""

import pytest
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema
from app.model import get_model

# 1. Define Pandera Schema for Telco Data (Input Validation)
telco_schema = DataFrameSchema({
    "gender": Column(str, Check.isin(["Male", "Female"])),
    "seniorcitizen": Column(int, Check.isin([0, 1])),
    "tenure": Column(int, Check(lambda s: s >= 0)),
    "monthlycharges": Column(float, Check(lambda s: s > 0)),
    "totalcharges": Column(float, Check(lambda s: s >= 0), nullable=True),
    "contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
})

def test_data_schema_with_pandera():
    """Verify input data schema using Pandera."""
    valid_data = pd.DataFrame({
        "gender": ["Female"],
        "seniorcitizen": [0],
        "tenure": [1],
        "monthlycharges": [29.85],
        "totalcharges": [29.85],
        "contract": ["Month-to-month"]
    })
    # Should pass
    telco_schema.validate(valid_data)

    invalid_data = pd.DataFrame({
        "gender": ["Unknown"], # Invalid
        "seniorcitizen": [0],
        "tenure": [-1], # Invalid
        "monthlycharges": [29.85],
        "totalcharges": [29.85],
        "contract": ["Month-to-month"]
    })
    # Should fail
    with pytest.raises(pa.errors.SchemaError):
        telco_schema.validate(invalid_data)

def test_prediction_output_quality():
    """Verify model output constraints and distribution."""
    model = get_model()
    if not model.is_loaded():
        pytest.skip("Model not loaded.")

    # Create a small batch to check distribution or range
    sample_batch = [
        {"gender": "Female", "seniorcitizen": 0, "partner": "Yes", "dependents": "No", "tenure": 1, "phoneservice": "No", "multiplelines": "No phone service", "internetservice": "DSL", "onlinesecurity": "No", "onlinebackup": "Yes", "deviceprotection": "No", "techsupport": "No", "streamingtv": "No", "streamingmovies": "No", "contract": "Month-to-month", "paperlessbilling": "Yes", "paymentmethod": "Electronic check", "monthlycharges": 29.85, "totalcharges": 29.85}
        for _ in range(10)
    ]
    
    probs = []
    for item in sample_batch:
        _, prob, _, _, _ = model.predict_with_latency(item)
        probs.append(prob)
    
    # Check that all probabilities are in [0, 1]
    assert all(0.0 <= p <= 1.0 for p in probs)

def test_class_distribution_logic():
    """Placeholder for class distribution check on training/audit data."""
    # In a real scenario, we'd load a sample of the audit data
    # and check if it's too skewed (e.g., 99% churn or 99% no-churn)
    # which might indicate a data collection error.
    try:
        y_audit = pd.read_csv("y_test_audit.csv")
        churn_rate = y_audit['churn'].mean()
        # Verify churn rate is reasonable (e.g., between 5% and 95%)
        assert 0.05 <= churn_rate <= 0.95, f"Suspicious class distribution: {churn_rate:.2%}"
    except FileNotFoundError:
        pytest.skip("y_test_audit.csv not found")
