"""
PREMIUM CHURN TRAINING PIPELINE (MULTI-EXPERIMENT)
Optimized for: Local (Mac Silicon) & Google Colab
Features: Anti-Leakage, Hyperparameter Tuning, SHAP Explainability, DagsHub Integration
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Machine Learning
import mlflow
import mlflow.sklearn
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from mlflow.models import infer_signature

# Explainability & Data Download
try:
    import shap
    import opendatasets as od
    from dotenv import load_dotenv
except ImportError:
    print("Installing missing libraries...")
    os.system("pip install shap opendatasets python-dotenv -q")
    import shap
    import opendatasets as od
    from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. SETUP & CREDENTIALS
# ---------------------------------------------------------
load_dotenv(override=True)
if os.path.exists(".secret"):
    load_dotenv(".secret", override=True)

def setup_mlflow():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
        print(f"🔗 Connected to MLflow Tracking Server: {tracking_uri}")
    else:
        print("⚠️ Warning: No tracking URI found. Logging locally.")

# ---------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------
def get_dataset():
    dataset_url = "https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset"
    data_file = "customer-churn-dataset/customer_churn_dataset-training-master.csv"
    
    if not os.path.exists(data_file):
        print("🚀 Downloading dataset from Kaggle...")
        kaggle_creds = {
            "username": os.getenv("KAGGLE_USERNAME"),
            "key": os.getenv("KAGGLE_KEY")
        }
        with open("kaggle.json", "w") as f:
            json.dump(kaggle_creds, f)
        
        od.download(dataset_url)
        if os.path.exists("kaggle.json"):
            os.remove("kaggle.json")
            
    df = pd.read_csv(data_file)
    print(f"✅ Data loaded: {len(df)} rows")
    return df

# ---------------------------------------------------------
# 3. EXPERIMENT RUNNER
# ---------------------------------------------------------
def run_model_experiment(df, model_type="xgboost"):
    # Pre-processing & Anti-Leakage
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Drop columns that cause Data Leakage (last_interaction) or aren't features (customerid)
    cols_to_drop = [c for c in ['customerid', 'last_interaction'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Numeric conversion safety
    if 'total_spend' in df.columns:
        df['total_spend'] = pd.to_numeric(df['total_spend'], errors='coerce')
    df = df.dropna(subset=['churn']) # Drop missing labels
    
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocessing Pipeline
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])

    # Model Selection
    if model_type == "xgboost":
        cw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        clf = XGBClassifier(scale_pos_weight=cw, eval_metric='logloss', random_state=42)
        params = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 5]}
    else:
        clf = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
        params = {'classifier__C': [0.1, 1.0]}

    pipeline = Pipeline([('pre', preprocessor), ('classifier', clf)])
    
    # Tracking
    with mlflow.start_run(run_name=f"Final_{model_type.upper()}"):
        print(f"🔥 Training {model_type}...")
        search = RandomizedSearchCV(pipeline, params, n_iter=3, cv=3, scoring='recall', n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Metrics
        y_pred = best_model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        print(f"      Stats -> Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}")

        # Visuals: Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix ({model_type})")
        cm_path = f"cm_{model_type}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # SHAP (Explainability)
        try:
            sample = X_test.sample(100)
            encoded_data = best_model.named_steps['pre'].transform(sample)
            feature_names = best_model.named_steps['pre'].get_feature_names_out()
            trans_df = pd.DataFrame(encoded_data, columns=feature_names)
            
            explainer = shap.Explainer(best_model.named_steps['classifier'])
            shap_values = explainer(trans_df)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, trans_df, show=False)
            plt.tight_layout()
            shap_path = f"shap_summary_{model_type}.png"
            plt.savefig(shap_path)
            mlflow.log_artifact(shap_path)
            plt.close()
            print("      SHAP artifact logged.")
        except Exception as e:
            print(f"      ⚠️ SHAP failed: {e}")

        # Model Registry
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        print(f"✅ Model {model_type} registered.")

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    setup_mlflow()
    df = get_dataset()
    for model_type in ["xgboost", "logistic_regression"]:
        run_model_experiment(df, model_type)
    print("\n🏁 ALL EXPERIMENTS FINISHED SUCCESSFULLY")
