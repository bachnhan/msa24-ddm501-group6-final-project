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

    # Synthetic Telco-style test data
    data_size = 500
    data = {
        "gender": np.random.choice(["Male", "Female"], data_size),
        "seniorcitizen": np.random.choice([0, 1], data_size),
        "partner": np.random.choice(["Yes", "No"], data_size),
        "dependents": np.random.choice(["Yes", "No"], data_size),
        "tenure": np.random.randint(1, 72, data_size),
        "phoneservice": np.random.choice(["Yes", "No"], data_size),
        "multiplelines": np.random.choice(["No phone service", "No", "Yes"], data_size),
        "internetservice": np.random.choice(["DSL", "Fiber optic", "No"], data_size),
        "onlinesecurity": np.random.choice(["No", "Yes", "No internet service"], data_size),
        "onlinebackup": np.random.choice(["No", "Yes", "No internet service"], data_size),
        "deviceprotection": np.random.choice(["No", "Yes", "No internet service"], data_size),
        "techsupport": np.random.choice(["No", "Yes", "No internet service"], data_size),
        "streamingtv": np.random.choice(["No", "Yes", "No internet service"], data_size),
        "streamingmovies": np.random.choice(["No", "Yes", "No internet service"], data_size),
        "contract": np.random.choice(["Month-to-month", "One year", "Two year"], data_size),
        "paperlessbilling": np.random.choice(["Yes", "No"], data_size),
        "paymentmethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], data_size),
        "monthlycharges": np.random.uniform(20, 120, data_size),
        "totalcharges": np.random.uniform(20, 8000, data_size),
        "churn": np.random.choice([0, 1], data_size)
    }
    df = pd.DataFrame(data)
    
    X = df.drop('churn', axis=1)
    
    # Get predictions
    probs = model.predict_proba(X)[:, 1]
    df['prob'] = probs
    df['error'] = np.abs(df['churn'] - df['prob'])
    
    # Compare Gender Groups
    male_res = df[df['gender'] == 'Male']['error'].mean()
    female_res = df[df['gender'] == 'Female']['error'].mean()
    
    print("\n" + "="*40)
    print("FAIRNESS ANALYSIS REPORT (Gender Bias)")
    print("="*40)
    print(f"Group: Male   | MAE: {male_res:.4f}")
    print(f"Group: Female | MAE: {female_res:.4f}")
    print("-" * 20)
    
    gap = np.abs(male_res - female_res)
    print(f"Bias Gap: {gap:.4f}")
    
    if gap < 0.05:
        print("Conclusion: Model shows parity across gender groups.")
    else:
        print("Warning: Bias detected. Consider re-balancing training data.")

if __name__ == "__main__":
    run_fairness_analysis()
