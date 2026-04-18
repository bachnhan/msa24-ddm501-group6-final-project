import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from dotenv import load_dotenv

# Load local .env file
load_dotenv()

def main():
    """Train Customer Churn model with MLflow tracking."""
    # Configure DagsHub Tracking
    repo_owner = "nhannhb92"
    repo_name = "msa24-ddm501-group6-final-project"
    model_name = os.getenv("MLFLOW_MODEL_NAME", "CustomerChurnModel")
    
    if os.getenv("MLFLOW_TRACKING_URI"):
        import dagshub
        print(f"[INFO] Initializing DagsHub connection...")
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
    print("=" * 60)
    print("Customer Churn Model Training with MLflow")
    print("=" * 60)
    
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "churn_model.pkl"
    
    # MLflow tracking
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    with mlflow.start_run():
        # Load Sample Data
        print("\n[1/4] Loading Customer Churn dataset...")
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
        
        # Preprocessing
        numeric_features = ['age', 'tenure', 'usage_frequency', 'support_calls', 'payment_delay', 'total_spend', 'last_interaction']
        categorical_features = ['gender', 'subscription_type', 'contract_length']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
        # Pipeline
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
        ])
        
        print("\n[2/4] Training RandomForest model...")
        model_pipeline.fit(X_train, y_train)
        
        # Eval
        y_pred = model_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        print(f"      Accuracy: {acc:.4f}")
        
        # Save & Register
        print(f"\n[3/4] Saving model to {model_path} and Registry...")
        with open(model_path, 'wb') as f:
            pickle.dump(model_pipeline, f)
            
        # Register the model in MLflow Registry
        mlflow.sklearn.log_model(
            sk_model=model_pipeline,
            artifact_path="model",
            registered_model_name=model_name
        )
        print(f"      Model registered as '{model_name}'")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
