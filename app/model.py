"""
ML Model wrapper for movie rating prediction with metrics instrumentation.


"""

import pickle
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional

from app.config import MODEL_PATH, MIN_RATING, MAX_RATING, MODEL_VERSION
from app.metrics import (
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    PREDICTION_VALUE,
    PREDICTION_ERRORS,
    MODEL_LOADED,
    MODEL_INFO,
    MODEL_LAST_RELOAD,
    BATCH_SIZE,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieRatingModel:
    """
    Wrapper class for the movie rating prediction model.
    
    This class handles:
    - Loading the trained model from disk
    - Making single predictions
    - Making batch predictions
    - Recording metrics for monitoring
    """
    
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the model wrapper.
        
        Args:
            model_path: Path to the saved model file (.pkl)
        """
        self.model_path = model_path
        self.model = None
        self.version = MODEL_VERSION
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model from disk and update metrics."""
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
            # Update model metrics after loading
            # Set MODEL_LOADED gauge to 1
            if MODEL_LOADED is not None:
                MODEL_LOADED.set(1)
            
            # Record the reload timestamp
            if MODEL_LAST_RELOAD is not None:
                MODEL_LAST_RELOAD.set(time.time())
            
            # Set model info
            if MODEL_INFO is not None:
                MODEL_INFO.info({
                    'version': self.version,
                    'type': 'SVD',
                    'path': str(self.model_path)
                })
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            
            # Set MODEL_LOADED to 0 on failure
            if MODEL_LOADED is not None:
                MODEL_LOADED.set(0)
            
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            
            # if MODEL_LOADED is not None:
            #     MODEL_LOADED.set(0)
            
            raise
    
    def predict(self, user_id: str, movie_id: str) -> float:
        """
        Predict rating for a single user-movie pair.
        
        Args:
            user_id: User ID (string)
            movie_id: Movie ID (string)
            
        Returns:
            Predicted rating (float between 1.0 and 5.0)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Record start time for latency measurement
        start_time = time.time()
        
        try:
            # Make prediction
            prediction = self.model.predict(user_id, movie_id)
            rating = round(prediction.est, 2)
            
            # Clip to valid range
            rating = max(MIN_RATING, min(MAX_RATING, rating))
            
            # Calculate duration and record metrics
            duration = time.time() - start_time
            
            # Record prediction count
            if PREDICTION_COUNT is not None:
                PREDICTION_COUNT.labels(model_version=self.version).inc()
            
            # Record prediction latency
            if PREDICTION_LATENCY is not None:
                PREDICTION_LATENCY.labels(model_version=self.version).observe(duration)
            
            # Record prediction value distribution
            if PREDICTION_VALUE is not None:
                PREDICTION_VALUE.labels(model_version=self.version).observe(rating)
            
            return rating
            
        except ValueError as e:
            # Record validation errors
            if PREDICTION_ERRORS is not None:
                PREDICTION_ERRORS.labels(
                    error_type='validation_error',
                    model_version=self.version
                ).inc()
            raise
            
        except Exception as e:
            # Record unknown errors
            if PREDICTION_ERRORS is not None:
                PREDICTION_ERRORS.labels(
                    error_type='unknown_error',
                    model_version=self.version
                ).inc()
            raise
    
    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict ratings for multiple user-movie pairs.
        
        Args:
            pairs: List of (user_id, movie_id) tuples
            
        Returns:
            List of predicted ratings
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Record batch size
        if BATCH_SIZE is not None:
            BATCH_SIZE.observe(len(pairs))
        
        return [self.predict(user_id, movie_id) for user_id, movie_id in pairs]
    
    def predict_with_latency(self, user_id: str, movie_id: str) -> Tuple[float, float]:
        """
        Predict rating and return latency in milliseconds.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Tuple of (predicted_rating, latency_ms)
        """
        start_time = time.time()
        rating = self.predict(user_id, movie_id)
        latency_ms = (time.time() - start_time) * 1000
        return rating, latency_ms
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_info(self) -> dict:
        """Get model information."""
        return {
            "version": self.version,
            "type": "SVD",
            "is_loaded": self.is_loaded(),
            "path": self.model_path,
        }


# =============================================================================
# Singleton instance management
# =============================================================================

_model_instance: Optional[MovieRatingModel] = None


def get_model() -> MovieRatingModel:
    """Get or create the model singleton instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = MovieRatingModel()
    return _model_instance


def reset_model() -> None:
    """Reset the model instance (useful for testing)."""
    global _model_instance
    _model_instance = None


def reload_model() -> MovieRatingModel:
    """Force reload the model."""
    global _model_instance
    _model_instance = MovieRatingModel()
    return _model_instance
