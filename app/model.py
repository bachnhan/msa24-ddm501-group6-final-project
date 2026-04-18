import os
from dotenv import load_dotenv

# Load local .env file (force override to ignore old terminal exports)
load_dotenv(override=True)

import pickle
import gzip
import time
import logging
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

from app.config import MODEL_PATH, MODEL_VERSION
from app.metrics import (
    MODEL_LOADED, 
    MODEL_LAST_RELOAD, 
    MODEL_INFO
)

logger = logging.getLogger(__name__)

class ChurnModel:
    """Predictor class for customer churn."""
    
    def __init__(self, model_path: str = MODEL_PATH):
        # We now expect a .gz file
        self.model_path = str(model_path) + ".gz" if not str(model_path).endswith(".gz") else str(model_path)
        self.model = None
        self.version = MODEL_VERSION
        self.last_error = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model from disk or MLflow Registry."""
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        model_name = os.getenv("MLFLOW_MODEL_NAME", "CustomerChurnModel")
        
        try:
            if tracking_uri and username and password:
                os.environ['MLFLOW_TRACKING_USERNAME'] = username
                os.environ['MLFLOW_TRACKING_PASSWORD'] = password
                logger.info(f"Connecting to MLflow Registry: {tracking_uri}")
                mlflow.set_tracking_uri(tracking_uri)
                
                # Fetching the 'Champion' or latest version
                # Note: Registry usually stores uncompressed. 
                # For this specific project flow, we prioritize the local path which we push to Git.
                model_uri = f"models:/{model_name}/latest"
                try:
                    self.model = mlflow.sklearn.load_model(model_uri)
                    logger.info(f"Successfully loaded model from Registry: {model_uri}")
                except Exception as reg_err:
                    logger.warning(f"Registry load failed, falling back to local compressed archive: {reg_err}")
                    raise reg_err
            else:
                raise ValueError("Registry credentials missing")

        except Exception as e:
            # Fallback to local compressed pickle
            try:
                if os.path.exists(self.model_path):
                    with gzip.open(self.model_path, "rb") as f:
                        self.model = pickle.load(f)
                    logger.info(f"Loaded compressed model from local path: {self.model_path}")
                else:
                    # Try original path if .gz doesn't exist
                    orig_path = self.model_path.replace(".gz", "")
                    with open(orig_path, "rb") as f:
                        self.model = pickle.load(f)
                    logger.info(f"Loaded uncompressed model from local path: {orig_path}")
            except Exception as local_err:
                self.last_error = f"Registry error: {e}. Local error: {local_err}"
                logger.error(f"Error loading model: {self.last_error}")
                if MODEL_LOADED is not None:
                    MODEL_LOADED.set(0)
                return

        self.last_error = None
        if MODEL_LOADED is not None:
            MODEL_LOADED.set(1)
    
    def get_last_error(self) -> Optional[str]:
        return self.last_error
    
    def predict(self, data_dict: dict) -> Tuple[bool, float]:
        """Make a churn prediction."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        X = pd.DataFrame([data_dict])
        churn_prob = float(self.model.predict_proba(X)[0, 1])
        is_churn = bool(self.model.predict(X)[0])
        return is_churn, churn_prob

    def is_loaded(self) -> bool:
        return self.model is not None

# Singleton instance
_model_instance = None

def get_model() -> ChurnModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = ChurnModel()
    return _model_instance
