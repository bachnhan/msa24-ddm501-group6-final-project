"""
FastAPI application for Movie Rating Prediction with Prometheus Monitoring.

This application exposes:
- /predict - Make predictions
- /health - Health check
- /metrics - Prometheus metrics endpoint
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import logging

from app.config import (
    API_TITLE, 
    API_DESCRIPTION, 
    API_VERSION, 
    MODEL_VERSION,
    METRICS_ENABLED,
)
from app.model import MovieRatingModel
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    MetricsInfo,
)
from app.middleware import MetricsMiddleware
from app.metrics import count_implemented_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add metrics middleware
if METRICS_ENABLED:
    app.add_middleware(MetricsMiddleware)

# Global model instance
model: MovieRatingModel = None


@app.on_event("startup")
async def startup_event():
    """Load model when application starts."""
    global model
    try:
        model = MovieRatingModel()
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


# =============================================================================
# Info Endpoints
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    implemented, total = count_implemented_metrics()
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "metrics_implemented": f"{implemented}/{total}",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the API and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy" if model and model.is_loaded() else "unhealthy",
        model_loaded=model is not None and model.is_loaded(),
        model_version=MODEL_VERSION
    )


# =============================================================================
# Metrics Endpoint
# =============================================================================

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns all collected metrics in Prometheus text format.
    This endpoint should be scraped by Prometheus.
    """
    if not METRICS_ENABLED:
        raise HTTPException(status_code=503, detail="Metrics disabled")
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/metrics/info", response_model=MetricsInfo, tags=["Monitoring"])
async def metrics_info():
    """Get information about metrics configuration."""
    implemented, total = count_implemented_metrics()
    return MetricsInfo(
        metrics_enabled=METRICS_ENABLED,
        endpoint="/metrics",
        metrics_count=implemented
    )


# =============================================================================
# Prediction Endpoints
# =============================================================================

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict movie rating for a user.
    
    Args:
        request: PredictionRequest with user_id and movie_id
        
    Returns:
        PredictionResponse with predicted rating and latency
    """
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        rating, latency_ms = model.predict_with_latency(
            request.user_id, 
            request.movie_id
        )
        return PredictionResponse(
            user_id=request.user_id,
            movie_id=request.movie_id,
            predicted_rating=rating,
            model_version=MODEL_VERSION,
            latency_ms=round(latency_ms, 3)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict movie ratings for multiple user-movie pairs.
    
    Args:
        request: BatchPredictionRequest with list of predictions
        
    Returns:
        BatchPredictionResponse with all predicted ratings
    """
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        total_latency = 0
        
        for item in request.predictions:
            rating, latency_ms = model.predict_with_latency(
                item.user_id, 
                item.movie_id
            )
            total_latency += latency_ms
            results.append(PredictionResponse(
                user_id=item.user_id,
                movie_id=item.movie_id,
                predicted_rating=rating,
                model_version=MODEL_VERSION,
                latency_ms=round(latency_ms, 3)
            ))
        
        avg_latency = total_latency / len(results) if results else 0
        
        return BatchPredictionResponse(
            predictions=results,
            total_count=len(results),
            avg_latency_ms=round(avg_latency, 3)
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Model Info Endpoint
# =============================================================================

@app.get("/model/info", tags=["Info"])
async def model_info():
    """Get information about the loaded model."""
    if model is None:
        return {
            "model_version": MODEL_VERSION,
            "model_type": "SVD (Collaborative Filtering)",
            "is_loaded": False,
        }
    return model.get_info()


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
