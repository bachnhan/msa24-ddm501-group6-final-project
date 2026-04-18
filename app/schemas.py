from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class PredictionRequest(BaseModel):
    """Input features for customer churn prediction."""
    age: int = Field(..., example=30, description="Customer age")
    gender: str = Field(..., example="Male", description="Customer gender (Male/Female)")
    tenure: int = Field(..., example=12, description="Number of months customer has been with the company")
    usage_frequency: int = Field(..., example=15, description="Frequency of app/service usage")
    support_calls: int = Field(..., example=2, description="Number of support calls made")
    payment_delay: int = Field(..., example=1, description="Number of payment delays")
    subscription_type: str = Field(..., example="Standard", description="Subscription level (Basic/Standard/Premium)")
    contract_length: str = Field(..., example="Monthly", description="Contract type (Monthly/Annual)")
    total_spend: float = Field(..., example=500.50, description="Total amount spent by customer")
    last_interaction: int = Field(..., example=5, description="Days since last interaction")

class PredictionResponse(BaseModel):
    """Output of the churn prediction."""
    model_config = ConfigDict(protected_namespaces=())
    churn_probability: float = Field(..., example=0.15)
    is_churn: bool = Field(..., example=False)
    model_version: str = Field(..., example="1.0.0")
    latency_ms: float = Field(..., example=12.5)

class HealthResponse(BaseModel):
    """Health check response."""
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_version: str
    error: Optional[str] = None

class MetricsInfo(BaseModel):
    """Metrics configuration info."""
    metrics_enabled: bool
    endpoint: str
    metrics_count: int

class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    predictions: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total_count: int
    avg_latency_ms: float
