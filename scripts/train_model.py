from dotenv import load_dotenv
import os
import pickle
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
from mlflow.models import infer_signature

# Load environment variables (Local .env, .secret, or Render Secret Files)
load_dotenv(override=True)
for secret_file in [".secret", "/etc/secrets/.env", "/etc/secrets/.secret"]:
    if os.path.exists(secret_file):
        load_dotenv(secret_file, override=True)

def main():
    """
    Train Customer Churn model and register it with MLflow Model Registry.
    This script is designed to run in environments with stable internet (like Google Colab).
    """
    # Environment Configuration
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "CustomerChurnModel")
    
    print("=" * 60)
    print("Customer Churn Model Training Pipeline")
    print("=" * 60)
    
    # Configure MLflow Tracking
    if tracking_uri and username and password:
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[INFO] Using Remote Tracking: {tracking_uri}")
    else:
        print("[WARNING] Remote Tracking credentials not found. Logging locally.")

    # Prepare local storage
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "churn_model.pkl"
    
    # Start MLflow Experiment
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    with mlflow.start_run() as run:
        # 1. Generate/Load Training Data
        print("\n[1/4] Preparing dataset...")
        data_size = 1000
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. Define Preprocessing & Pipeline
        numeric_features = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'total_spend', 'last_interaction']
        categorical_features = ['gender', 'subscription_type', 'contract_length']
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])
        
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
        ])
        
        # 3. Train and Evaluate
        print(f"\n[2/4] Training RandomForest model (Estimators: 100)...")
        model_pipeline.fit(X_train, y_train)
        
        y_pred = model_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        print(f"      Validation Accuracy: {acc:.4f}")
        
        # 4. Save and Register
        print(f"\n[3/4] Registering model to Registry as '{model_name}'...")
        
        # Save local copy for Git/Fallback
        with open(model_path, 'wb') as f:
            pickle.dump(model_pipeline, f)
            
        # Infer signature and create input example for better UI display
        signature = infer_signature(X_train, model_pipeline.predict(X_train))
        input_example = X_train.iloc[:3]
        
        # Register in MLflow Registry with full metadata
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example
        )
        print(f"      Successfully Registered Version in Registry.")

    print("\n" + "=" * 60)
    print("Training Pipeline Finished Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
