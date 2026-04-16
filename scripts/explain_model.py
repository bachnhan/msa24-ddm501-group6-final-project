"""
Explainability analysis for the Customer Churn Prediction System.
Uses feature importance from the RandomForest model.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def explain_churn(model_path="models/churn_model.pkl"):
    if not Path(model_path).exists():
        print("Model not found.")
        return

    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # Get feature names after preprocessing
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Extract feature names
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['gender', 'subscription_type', 'contract_length'])
    num_features = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'total_spend', 'last_interaction']
    feature_names = np.concatenate([num_features, cat_features])
    
    importances = classifier.feature_importances_
    results = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    results = results.sort_values(by='Importance', ascending=False)
    
    print("\n" + "="*40)
    print("FEATURE IMPORTANCE (GLOBAL EXPLANATION)")
    print("="*40)
    print(results.head(10))
    print("="*40)
    print("\nInterpretation:")
    top_feature = results.iloc[0]['Feature']
    print(f"The most critical factor driving customer churn is: {top_feature}")

if __name__ == "__main__":
    explain_churn()
