"""
Configuration settings for the Customer Churn API.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "churn_model.pkl"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# API settings
API_TITLE = "Customer Churn Prediction API"
API_DESCRIPTION = "API for predicting customer churn with Prometheus monitoring"
API_VERSION = "1.0.0"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Prediction constraints (Churn probability 0.0 - 1.0)
MIN_PROB = 0.0
MAX_PROB = 1.0

# Monitoring settings
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"
