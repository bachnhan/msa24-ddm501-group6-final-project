"""
ML Model wrapper for Customer Churn Prediction.
Uses a scikit-learn pipeline for classification.
"""

import pickle
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

from app.config import MODEL_PATH, MODEL_VERSION
from app.metrics import (
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    PREDICTION_VALUE,
    PREDICTION_ERRORS,
    MODEL_LOADED,
    MODEL_INFO,
    MODEL_LAST_RELOAD,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModel:
    """
    Wrapper class for the Customer Churn Prediction model.
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.version = MODEL_VERSION
        self.last_error = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model from disk."""
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Churn Model loaded successfully from {self.model_path}")
            self.last_error = None
            
            if MODEL_LOADED is not None:
                MODEL_LOADED.set(1)
            if MODEL_LAST_RELOAD is not None:
                MODEL_LAST_RELOAD.set(time.time())
            if MODEL_INFO is not None:
                MODEL_INFO.info({
                    'version': self.version,
                    'type': 'RandomForestClassifier',
                    'path': str(self.model_path)
                })
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error loading model: {e}")
            if MODEL_LOADED is not None:
                MODEL_LOADED.set(0)
            # We don't raise here to allow the API to start in a degraded state
    
    def get_last_error(self) -> Optional[str]:
        return self.last_error
    
    def predict(self, data_dict: dict) -> Tuple[bool, float]:
        """
        Make a churn prediction for a single customer.
        
        Returns:
            Tuple of (is_churn, churn_probability)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        try:
            # Convert dictionary to DataFrame for the pipeline
            df = pd.DataFrame([data_dict])
            
            # Get probability [class_0_prob, class_1_prob]
            probs = self.model.predict_proba(df)[0]
            churn_prob = float(probs[1])
            is_churn = bool(churn_prob >= 0.5)
            
            duration = time.time() - start_time
            
            # Record metrics
            if PREDICTION_COUNT is not None:
                PREDICTION_COUNT.labels(model_version=self.version).inc()
            if PREDICTION_LATENCY is not None:
                PREDICTION_LATENCY.labels(model_version=self.version).observe(duration)
            if PREDICTION_VALUE is not None:
                PREDICTION_VALUE.labels(model_version=self.version).observe(churn_prob)
            
            return is_churn, churn_prob
            
        except Exception as e:
            if PREDICTION_ERRORS is not None:
                PREDICTION_ERRORS.labels(
                    error_type=type(e).__name__,
                    model_version=self.version
                ).inc()
            raise
    
    def predict_with_latency(self, data_dict: dict) -> Tuple[bool, float, float]:
        """Predict and return result with latency in ms."""
        start_time = time.time()
        is_churn, prob = self.predict(data_dict)
        latency_ms = (time.time() - start_time) * 1000
        return is_churn, prob, latency_ms

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_info(self) -> dict:
        return {
            "version": self.version,
            "type": "RandomForestClassifier",
            "is_loaded": self.is_loaded(),
            "path": self.model_path,
        }

# Singleton instance
_model_instance: Optional[ChurnModel] = None

def get_model() -> ChurnModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = ChurnModel()
    return _model_instance
