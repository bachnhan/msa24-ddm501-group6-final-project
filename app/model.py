import os
import pickle
import time
import logging
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables (Local .env, .secret, or Render Secret Files)
load_dotenv(override=True)
for secret_file in [".secret", "/etc/secrets/.env", "/etc/secrets/.secret"]:
    if os.path.exists(secret_file):
        load_dotenv(secret_file, override=True)

from app.config import MODEL_PATH, MODEL_VERSION
from app.metrics import (
    MODEL_LOADED, 
    MODEL_LAST_RELOAD, 
    MODEL_INFO
)

logger = logging.getLogger(__name__)

class ChurnModel:
    """Predictor class for customer churn using MLflow Registry."""
    
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.version = MODEL_VERSION
        self.last_error = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the trained model.
        Primary: DagsHub MLflow Model Registry (Champion/Latest).
        Fallback: Local pickle file from the models/ directory.
        """
        import mlflow
        
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        model_name = os.getenv("MLFLOW_MODEL_NAME", "CustomerChurnModel")
        
        try:
            if tracking_uri and username and (password or os.path.exists("/etc/secrets/.env") or os.path.exists("/etc/secrets/.secret")):
                # Ensure we have the password from secret files if dashboard is empty
                if not password:
                    for sf in ["/etc/secrets/.env", "/etc/secrets/.secret"]:
                        if os.path.exists(sf):
                            load_dotenv(sf, override=True)
                    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
                
                # Set authentication for MLflow client
                os.environ['MLFLOW_TRACKING_USERNAME'] = username
                os.environ['MLFLOW_TRACKING_PASSWORD'] = password
                
                logger.info(f"Connecting to MLflow Tracking Server: {tracking_uri}")
                mlflow.set_tracking_uri(tracking_uri)

                # Flexible reference: can be a version number (1, 2) or an alias (Champion, Production)
                model_ref = os.getenv("MLFLOW_MODEL_VERSION", "latest")
                
                # If ref is purely numeric, it's a version number, use / syntax
                if model_ref.isdigit():
                    model_uri = f"models:/{model_name}/{model_ref}"
                else:
                    # Otherwise, it's an alias, use @ syntax
                    model_uri = f"models:/{model_name}@{model_ref}"
                    
                logger.info(f"Fetching model from Registry using URI: {model_uri}")
                self.model = mlflow.sklearn.load_model(model_uri)
                logger.info("Successfully loaded model from MLflow Registry.")
            else:
                logger.warning("MLflow credentials missing. Falling back to local model.")
                raise ValueError("Missing MLflow environment variables.")

        except Exception as e:
            logger.error(f"Failed to load from Registry: {e}")
            # Fallback to local .pkl file
            try:
                if os.path.exists(self.model_path):
                    with open(self.model_path, "rb") as f:
                        self.model = pickle.load(f)
                    logger.info(f"Successfully loaded model from local path: {self.model_path}")
                else:
                    self.last_error = f"Model file not found at {self.model_path}"
                    logger.error(self.last_error)
            except Exception as local_err:
                self.last_error = f"Critical Error: {local_err}"
                logger.error(self.last_error)
        
        # Update metrics
        if self.model is not None:
            self.last_error = None
            if MODEL_LOADED is not None:
                MODEL_LOADED.set(1)
        else:
            if MODEL_LOADED is not None:
                MODEL_LOADED.set(0)
    
    def get_last_error(self) -> Optional[str]:
        """Return the last error message if model failed to load."""
        return self.last_error
    
    def predict(self, data_dict: dict) -> Tuple[bool, float]:
        """
        Process input features and return churn prediction.
        
        Args:
            data_dict (dict): Dictionary containing customer features.
            
        Returns:
            Tuple[bool, float]: (Is Churn, Probability of Churn)
        """
        is_churn, churn_prob, _ = self.predict_with_latency(data_dict)
        return is_churn, churn_prob

    def predict_with_latency(self, data_dict: dict) -> Tuple[bool, float, float]:
        """
        Predict churn and also return the performance latency in milliseconds.
        
        Returns:
            Tuple[bool, float, float]: (Is Churn, Probability, Latency MS)
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot perform prediction.")
            
        start_time = time.perf_counter()
        
        # Convert dictionary to DataFrame
        X = pd.DataFrame([data_dict])
        
        # Get predictions
        churn_prob = float(self.model.predict_proba(X)[0, 1])
        is_churn = bool(self.model.predict(X)[0])
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return is_churn, churn_prob, latency_ms

    def is_loaded(self) -> bool:
        """Check if the model instance is ready for prediction."""
        return self.model is not None

# Singleton Pattern for Model Instance
_model_instance = None

def get_model() -> ChurnModel:
    """Get the global ChurnModel singleton."""
    global _model_instance
    if _model_instance is None:
        _model_instance = ChurnModel()
    return _model_instance
