from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, 
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models import infer_signature
import shap

# Load configuration and secrets
load_dotenv(override=True)
for secret_file in [".secret", "/etc/secrets/.env", "/etc/secrets/.secret"]:
    if os.path.exists(secret_file):
        load_dotenv(secret_file, override=True)

def run_training_experiment(model_type="xgboost", run_name=None):
    repo_root = Path(__file__).parent.parent
    data_path = repo_root / "data" / "customer_churn_dataset-training-master.csv"
    model_name_base = os.getenv("MLFLOW_MODEL_NAME", "CustomerChurnModel")
    
    if run_name is None:
        run_name = f"{model_type.upper()}_Fast_Pipeline"

    with mlflow.start_run(run_name=run_name):
        print(f"\n🚀 [PIPELINE] Starting experiment: {run_name}")
        
        # 1. DATA INGESTION (Optimized)
        print("[1/6] Ingesting data & removing leakage...")
        df = pd.read_csv(data_path)
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        df = df.drop(columns=[c for c in ['customerid', 'last_interaction'] if c in df.columns])
        
        spend_cols = [c for c in ['total_spend', 'totalcharges'] if c in df.columns]
        for col in spend_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['churn'] + spend_cols)

        # 2. CORRELATION
        print("[2/6] Logging Correlation Matrix...")
        num_df = df.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=False, cmap='YlGnBu')
        plt.savefig("correlation.png")
        mlflow.log_artifact("correlation.png")
        plt.close()

        # 3. SPLIT
        X = df.drop('churn', axis=1)
        y = df['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 4. PREPROCESSING
        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline(steps=[('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), X.select_dtypes(include=[np.number]).columns.tolist()),
            ('cat', Pipeline(steps=[('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), X.select_dtypes(include=['object']).columns.tolist())
        ])

        # 5. TRAINING
        print(f"[3/6] Tuning and Training {model_type}...")
        if model_type == "xgboost":
            cw = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            clf = XGBClassifier(scale_pos_weight=cw, eval_metric='logloss', random_state=42)
            params = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [3, 5]}
        else:
            clf = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
            params = {'classifier__C': [0.1, 1.0]}

        search = RandomizedSearchCV(Pipeline([('pre', preprocessor), ('classifier', clf)]), params, n_iter=3, cv=3, scoring='recall', n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # 6. EVAL & VISUALS
        print("[4/6] Logging metrics and visuals...")
        y_pred = best_model.predict(X_test)
        mlflow.log_metrics({"accuracy": accuracy_score(y_test, y_pred), "recall": recall_score(y_test, y_pred)})
        
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.savefig(f"cm_{model_type}.png")
        mlflow.log_artifact(f"cm_{model_type}.png")
        plt.close()

        # 7. SHAP (Fast mode: 50 samples)
        print("[5/6] SHAP Explainability (Minimal sample)...")
        try:
            sample = X_test.sample(n=min(50, len(X_test)), random_state=42)
            trans_df = pd.DataFrame(best_model.named_steps['pre'].transform(sample), columns=best_model.named_steps['pre'].get_feature_names_out())
            explainer = shap.Explainer(best_model.named_steps['classifier'])
            shap_vals = explainer(trans_df)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, trans_df, show=False)
            plt.tight_layout()
            plt.savefig(f"shap_{model_type}.png")
            mlflow.log_artifact(f"shap_{model_type}.png")
            plt.close()
        except: pass

        # 8. REGISTRY
        print("[6/6] Finalizing Model Registration...")
        mlflow.sklearn.log_model(best_model, "model", registered_model_name=f"{model_name_base}_{model_type}")

def main():
    print("=" * 60)
    print("🚀 ULTRA-FAST PRODUCTION PIPELINE")
    print("=" * 60)
    
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
        os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    
    mlflow.set_experiment("Churn_Final_Ultra_Fast")
    
    for m in ["xgboost", "logistic_regression"]:
        try: run_training_experiment(m)
        except Exception as e: print(f"❌ Error: {e}")
    print("\n✅ Finished!")

if __name__ == "__main__":
    main()
