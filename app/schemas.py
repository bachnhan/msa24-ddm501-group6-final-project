from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Input features for Churn prediction (19 fields total matching Telco dataset)."""

    gender: str = Field(..., examples=["Female"])
    seniorcitizen: int = Field(..., examples=[0], ge=0, le=1)
    partner: str = Field(..., examples=["Yes"])
    dependents: str = Field(..., examples=["No"])
    tenure: int = Field(
        ..., examples=[1], ge=0, le=360, description="Tenure in months (0-360)"
    )
    phoneservice: str = Field(..., examples=["No"])
    multiplelines: str = Field(..., examples=["No phone service"])
    internetservice: str = Field(..., examples=["DSL"])
    onlinesecurity: str = Field(..., examples=["No"])
    onlinebackup: str = Field(..., examples=["Yes"])
    deviceprotection: str = Field(..., examples=["No"])
    techsupport: str = Field(..., examples=["No"])
    streamingtv: str = Field(..., examples=["No"])
    streamingmovies: str = Field(..., examples=["No"])
    contract: str = Field(..., examples=["Month-to-month"])
    paperlessbilling: str = Field(..., examples=["Yes"])
    paymentmethod: str = Field(..., examples=["Electronic check"])
    monthlycharges: float = Field(..., examples=[29.85], ge=0)
    totalcharges: float = Field(..., examples=[29.85], ge=0)


class PredictionResponse(BaseModel):
    """Refined output matching the requested sample with probability and risk tier."""

    model_config = ConfigDict(protected_namespaces=())
    churn_probability: float = Field(..., examples=[0.823])
    # Use lowercase examples for bool if preferred but here True is fine
    is_churn: bool = Field(..., examples=[True])
    risk_tier: str = Field(
        ..., examples=["High"], description="Risk level (Low, Medium, High)"
    )
    reason_codes: List[str] = Field(
        ...,
        examples=[["contract_type_monthly", "tenure_lt_12mo", "no_techsupport"]],
        description="Semantic slug-style reason codes",
    )
    model_version: str = Field(..., examples=["1.0.0"])


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
