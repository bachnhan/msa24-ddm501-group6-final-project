"""
Fairness and Bias Analysis for the Customer Churn Prediction System.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def run_fairness_analysis(model_path="models/churn_model.pkl"):
    if not Path(model_path).exists():
        print("Model not found.")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Synthetic test data
    data_size = 500
    data = {
        'age': np.random.randint(18, 70, data_size),
        'gender': np.random.choice(['Male', 'Female'], data_size),
        'tenure': np.random.randint(1, 72, data_size),
        'usage_frequency': np.random.randint(1, 31, data_size),
        'support_calls': np.random.randint(0, 10, data_size),
        'payment_delay': np.random.randint(0, 30, data_size),
        'subscription_type': np.random.choice(['Basic', 'Standard', 'Premium'], data_size),
        'contract_length': np.random.choice(['Monthly', 'Annual'], data_size),
        'total_spend': np.random.uniform(100, 5000, data_size),
        'last_interaction': np.random.randint(0, 30, data_size),
        'churn': np.random.choice([0, 1], data_size)
    }
    df = pd.DataFrame(data)
    
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Get predictions
    probs = model.predict_proba(X)[:, 1]
    df['prob'] = probs
    df['error'] = np.abs(df['churn'] - df['prob'])
    
    # Compare Gender Groups
    male_mae = df[df['gender'] == 'Male']['error'].mean()
    female_mae = df[df['gender'] == 'Female']['error'].mean()
    
    print("\n" + "="*40)
    print("FAIRNESS ANALYSIS REPORT (Gender Bias)")
    print("="*40)
    print(f"Group: Male   | MAE: {male_mae:.4f}")
    print(f"Group: Female | MAE: {female_mae:.4f}")
    print("-" * 20)
    
    gap = np.abs(male_mae - female_mae)
    print(f"Bias Gap: {gap:.4f}")
    
    if gap < 0.05:
        print("Conclusion: Model shows parity across gender groups.")
    else:
        print("Warning: Bias detected. Consider re-balancing training data.")

if __name__ == "__main__":
    run_fairness_analysis()
