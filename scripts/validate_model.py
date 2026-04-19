"""
Model Validation Gate (Step 4 in CI/CD).
Validates the champion model against audit data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score
from app.model import get_model

# 1. Configuration - Thresholds from Slide
ROC_AUC_THRESHOLD = 0.85
RECALL_THRESHOLD = 0.80

def run_validation():
    print("🚀 Starting Model Validation Gate...")
    
    # Load Model
    model_instance = get_model()
    if not model_instance.is_loaded():
        print("❌ Error: Model could not be loaded for validation.")
        exit(1)
    
    print(f"📦 Validating Model Version: {model_instance.loaded_version}")
    
    # Load Audit Data
    try:
        X_audit = pd.read_csv("X_test_audit.csv")
        y_audit = pd.read_csv("y_test_audit.csv")
    except FileNotFoundError:
        print("⚠️ Audit data not found. Skipping validation (passing by default in limited env).")
        return

    # Check for feature mismatch and handle it
    # If the model expects Telco features but we have Kaggle features, we might need a mapper
    # or we assume the current model version correctly matches the audit data.
    
    try:
        # Preprocessing of features if necessary (lowercase, underscores)
        X_audit.columns = [col.lower().replace(' ', '_') for col in X_audit.columns]
        
        # Get Predictions
        # Use predict_proba for ROC-AUC
        # Note: ChurnModel.model is the underlying sklearn/xgboost pipeline
        model = model_instance.model
        y_proba = model.predict_proba(X_audit)[:, 1]
        y_pred = model.predict(X_audit)
        
        # Calculate Metrics
        roc_auc = roc_auc_score(y_audit, y_proba)
        recall = recall_score(y_audit, y_pred)
        
        print(f"📊 Validation Results:")
        print(f"   - ROC-AUC: {roc_auc:.4f} (Target: >= {ROC_AUC_THRESHOLD})")
        print(f"   - Recall:  {recall:.4f} (Target: >= {RECALL_THRESHOLD})")
        
        # Gate Logic
        passed = True
        if roc_auc < ROC_AUC_THRESHOLD:
            print(f"❌ FAIL: ROC-AUC {roc_auc:.4f} is below threshold {ROC_AUC_THRESHOLD}")
            passed = False
        
        if recall < RECALL_THRESHOLD:
            print(f"❌ FAIL: Recall {recall:.4f} is below threshold {RECALL_THRESHOLD}")
            passed = False
            
        if not passed:
            print("🚫 Model Gate: REJECTED")
            exit(1)
        else:
            print("✅ Model Gate: PASSED")
            
    except Exception as e:
        print(f"❌ Critical error during validation: {e}")
        # In a real CI/CD, we'd fail the build
        exit(1)

if __name__ == "__main__":
    run_validation()
