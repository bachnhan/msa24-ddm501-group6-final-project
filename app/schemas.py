"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    
    user_id: str = Field(..., min_length=1, max_length=50, examples=["196"])
    movie_id: str = Field(..., min_length=1, max_length=50, examples=["242"])
    
    @field_validator("user_id", "movie_id")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate that IDs are not empty or whitespace only."""
        if not v.strip():
            raise ValueError("ID cannot be empty or whitespace only")
        return v.strip()


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    
    user_id: str
    movie_id: str
    predicted_rating: float = Field(..., ge=1.0, le=5.0)
    model_version: str
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    
    status: str
    model_loaded: bool
    model_version: str


class PredictionItem(BaseModel):
    """Single prediction item for batch requests."""
    
    user_id: str = Field(..., min_length=1, max_length=50)
    movie_id: str = Field(..., min_length=1, max_length=50)


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint."""
    
    predictions: List[PredictionItem] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint."""
    
    predictions: List[PredictionResponse]
    total_count: int
    avg_latency_ms: float


class MetricsInfo(BaseModel):
    """Information about available metrics."""
    
    metrics_enabled: bool
    endpoint: str
    metrics_count: int
