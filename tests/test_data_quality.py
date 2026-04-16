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
    # Invalid age type (should be int)
    with pytest.raises(Exception):
        PredictionRequest(
            age="thirty",
            gender="Male",
            tenure=12,
            usage_frequency=15,
            support_calls=2,
            payment_delay=1,
            subscription_type="Standard",
            contract_length="Monthly",
            total_spend=500.50,
            last_interaction=5
        )
    
    # Missing required field (tenure)
    with pytest.raises(Exception):
        PredictionRequest(
            age=30,
            gender="Male",
            usage_frequency=15,
            support_calls=2,
            payment_delay=1,
            subscription_type="Standard",
            contract_length="Monthly",
            total_spend=500.50,
            last_interaction=5
        )

def test_model_output_range():
    """Verify that model churn probability is within [0, 1]."""
    model_path = Path("models/svd_model.pkl")
    if not model_path.exists():
        pytest.skip("Model artifact not found. Run training first.")
        
    model = get_model()
    sample_data = {
        'age': 30, 'gender': 'Male', 'tenure': 12, 'usage_frequency': 15,
        'support_calls': 2, 'payment_delay': 1, 'subscription_type': 'Standard',
        'contract_length': 'Monthly', 'total_spend': 500.0, 'last_interaction': 5
    }
    is_churn, prob, _ = model.predict_with_latency(sample_data)
    
    assert 0.0 <= prob <= 1.0
    assert isinstance(is_churn, bool)

def test_data_completeness():
    """Verify essential project files exist."""
    assert Path("requirements.txt").exists()
    assert Path("app/main.py").exists()
    assert Path("docker-compose.yml").exists()
