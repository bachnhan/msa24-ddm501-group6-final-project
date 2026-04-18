from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class PredictionRequest(BaseModel):
    """Input features for Telco Customer Churn prediction."""
    gender: str = Field(..., example="Female")
    seniorcitizen: int = Field(..., example=0)
    partner: str = Field(..., example="Yes")
    dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=1)
    phoneservice: str = Field(..., example="No")
    multiplelines: str = Field(..., example="No phone service")
    internetservice: str = Field(..., example="DSL")
    onlinesecurity: str = Field(..., example="No")
    onlinebackup: str = Field(..., example="Yes")
    deviceprotection: str = Field(..., example="No")
    techsupport: str = Field(..., example="No")
    streamingtv: str = Field(..., example="No")
    streamingmovies: str = Field(..., example="No")
    contract: str = Field(..., example="Month-to-month")
    paperlessbilling: str = Field(..., example="Yes")
    paymentmethod: str = Field(..., example="Electronic check")
    monthlycharges: float = Field(..., example=29.85)
    totalcharges: float = Field(..., example=29.85)

class PredictionResponse(BaseModel):
    """Refined output matching the requested sample with probability and risk tier."""
    model_config = ConfigDict(protected_namespaces=())
    churn_probability: float = Field(..., example=0.823)
    is_churn: bool = Field(..., example=True)
    risk_tier: str = Field(..., example="High", description="Risk level (Low, Medium, High)")
    reason_codes: List[str] = Field(..., example=["contract_type_monthly", "tenure_lt_12mo", "no_techsupport"], description="Semantic slug-style reason codes")
    model_version: str = Field(..., example="1.0.0")

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
