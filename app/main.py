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
from app.model import get_model
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
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using Random Forest with Monitoring",
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

@app.get("/", tags=["Info"])
async def root():
    implemented, total = count_implemented_metrics()
    return {
        "name": "Customer Churn Prediction System",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    model = get_model()
    return HealthResponse(
        status="healthy" if model.is_loaded() else "unhealthy",
        model_loaded=model.is_loaded(),
        model_version=MODEL_VERSION,
        error=model.get_last_error() # New field for debugging
    )

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    if not METRICS_ENABLED:
        raise HTTPException(status_code=503, detail="Metrics disabled")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    model = get_model()
    if not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        data = request.dict()
        
        # --- API GUARDRAIL ---
        if data['age'] < 0 or data['age'] > 120:
            raise HTTPException(status_code=400, detail="Invalid age")
        
        is_churn, prob, latency_ms = model.predict_with_latency(data)
        
        return PredictionResponse(
            churn_probability=round(prob, 4),
            is_churn=is_churn,
            model_version=MODEL_VERSION,
            latency_ms=round(latency_ms, 3)
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    model = get_model()
    if not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        total_latency = 0
        
        for item in request.predictions:
            is_churn, prob, latency_ms = model.predict_with_latency(item.dict())
            total_latency += latency_ms
            results.append(PredictionResponse(
                churn_probability=round(prob, 4),
                is_churn=is_churn,
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

@app.post("/model/reload", tags=["Admin"])
async def reload_model(version_or_alias: str = "latest"):
    """
    Force the API to reload the model from DagsHub Registry.
    You can provide a version number (1, 2) or an alias (Production, Champion).
    """
    model = get_model()
    success = model.reload(version_or_alias)
    
    if success:
        return {
            "status": "success", 
            "message": f"Model reloaded successfully using reference: {version_or_alias}",
            "model_loaded": True
        }
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to reload model: {model.get_last_error()}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
