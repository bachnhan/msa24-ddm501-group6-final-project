import os
from dotenv import load_dotenv

# Load local .env file (force override to ignore old terminal exports)
load_dotenv(override=True)

import pickle
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def main():
    """Train model and save with GZIP compression to bypass network limits."""
    # Configure 
    repo_owner = "nhannhb92"
    repo_name = "msa24-ddm501-group6-final-project"
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    print("=" * 60)
    print("Training Model with GZIP Compression (MLOps)")
    print("=" * 60)
    
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "churn_model.pkl.gz"
    
    # [1/4] Training - Full Power (100 trees)
    print("\n[1/4] Training full RandomForest model (100 trees)...")
    data_size = 1000
    data = {'age': np.random.randint(18, 70, data_size), 'gender': np.random.choice(['Male', 'Female'], data_size), 'tenure': np.random.randint(1, 72, data_size), 'usage_frequency': np.random.randint(1, 31, data_size), 'support_calls': np.random.randint(0, 10, data_size), 'payment_delay': np.random.randint(0, 30, data_size), 'subscription_type': np.random.choice(['Basic', 'Standard', 'Premium'], data_size), 'contract_length': np.random.choice(['Monthly', 'Annual'], data_size), 'total_spend': np.random.uniform(100, 5000, data_size), 'last_interaction': np.random.randint(0, 30, data_size), 'churn': np.random.choice([0, 1], data_size)}
    df = pd.DataFrame(data)
    X = df.drop('churn', axis=1); y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', ColumnTransformer(transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'total_spend', 'last_interaction']),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), ['gender', 'subscription_type', 'contract_length'])
        ])),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    model_pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, model_pipeline.predict(X_test))
    print(f"      Accuracy: {acc:.4f}")
    
    # [2/4] Saving Local with GZIP
    print(f"\n[2/4] Saving compressed model to {model_path}...")
    with gzip.open(model_path, 'wb') as f:
        pickle.dump(model_pipeline, f)
    
    actual_size = os.path.getsize(model_path) / 1024
    print(f"      Compressed size: {actual_size:.1f} KB (Reduced from ~2200 KB)")
            
    # [3/4] Uploading
    print("\n[3/4] Uploading compressed model to DagsHub...")
    try:
        from dagshub.upload import Repo
        repo = Repo(repo_owner, repo_name, username=username, password=password)
        repo.upload(
            str(model_path),
            "models/churn_model.pkl.gz",
            "Update trained model artifact [COMPRESSED]"
        )
        print("      ✅ SUCCESS! The compressed model is in the Cloud.")
    except Exception as e:
        print(f"      ❌ Upload Failed: {e}")

    print("\n" + "=" * 60)
    print("Process Finished!")
    print("=" * 60)

if __name__ == "__main__":
    main()
