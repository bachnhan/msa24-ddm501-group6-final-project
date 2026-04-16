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

def main():
    """Train Customer Churn model with MLflow tracking."""
    
    print("=" * 60)
    print("Customer Churn Model Training with MLflow")
    print("=" * 60)
    
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "churn_model.pkl"
    
    # MLflow tracking
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    with mlflow.start_run():
        # Load Sample Data (Simulating the Kaggle dataset structure)
        print("\n[1/4] Loading Customer Churn dataset...")
        
        # Creating synthetic data to match the Kaggle schema for demonstration
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
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Pipeline
        n_estimators = 100
        max_depth = 10
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42))
        ])
        
        # Log params
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "numeric_features": len(numeric_features),
            "categorical_features": len(categorical_features)
        })
        
        print("\n[2/4] Training RandomForest model...")
        model_pipeline.fit(X_train, y_train)
        
        # Eval
        y_pred = model_pipeline.predict(X_test)
        y_prob = model_pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        mlflow.log_metrics({"accuracy": acc, "f1": f1, "auc": auc})
        print(f"      Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        # Save
        print(f"\n[3/4] Saving model to {model_path}...")
        with open(model_path, 'wb') as f:
            pickle.dump(model_pipeline, f)
            
        mlflow.log_artifact(str(model_path))
        print("      Model and artifacts logged successfully!")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
