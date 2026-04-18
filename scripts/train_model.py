import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import dagshub
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, f1_score, roc_curve, auc
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
from dotenv import load_dotenv

# Load Local Secrets
load_dotenv(".secret")

def clean_dataset(df):
    """Standard cleaning logic shared with API."""
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    # Drop leakage and ID columns
    df = df.drop(columns=[c for c in ['customerid', 'last_interaction'] if c in df.columns])
    # Guardrails
    if 'age' in df.columns:
        df = df[(df['age'] >= 18) & (df['age'] <= 120)]
    for col in [c for c in ['total_spend', 'totalcharges'] if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['churn'])

def train_and_register():
    # 1. Initialize DagsHub
    dagshub.init(repo_owner="nhannhb92", repo_name="msa24-ddm501-group6-final-project", mlflow=True)
    mlflow.sklearn.autolog(log_models=False)
    
    # 2. Data Loading
    print("📂 Loading data...")
    # In a real script, you'd download or use local paths. Assuming local csv for this script.
    try:
        df_raw = clean_dataset(pd.read_csv("customer-churn-dataset/customer_churn_dataset-training-master.csv"))
        df_test_final = clean_dataset(pd.read_csv("customer-churn-dataset/customer_churn_dataset-testing-master.csv"))
    except FileNotFoundError:
        print("⚠️ Local data not found. Please run download script first.")
        return

    df_train, df_val = train_test_split(df_raw, test_size=0.2, random_state=42, stratify=df_raw['churn'])
    
    # Data Profiling
    print("\n--- 📊 DATA PROFILE ---")
    print(f"Train samples: {len(df_train)} | Test samples: {len(df_test_final)}")
    print(f"Primary Churn Rate: {df_train['churn'].mean():.2%}")
    
    # 3. Features
    X_train = df_train.drop(columns=['churn'])
    y_train = df_train['churn']
    X_test = df_test_final.drop(columns=['churn'])
    y_test = df_test_final['churn']

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), X_train.select_dtypes(include=[np.number]).columns.tolist()),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), X_train.select_dtypes(include=['object']).columns.tolist())
    ])

    models_to_run = ["xgboost", "random_forest", "logistic_regression"]
    
    for model_type in models_to_run:
        mlflow.set_experiment("Churn_Production_Automation")
        with mlflow.start_run(run_name=f"SCRIPT_{model_type.upper()}"):
            print(f"\n🚀 Training {model_type.upper()}...")
            
            if model_type == "xgboost":
                pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                clf = XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42)
                params = {'clf__max_depth': [3, 5]}
            elif model_type == "random_forest":
                clf = RandomForestClassifier(class_weight='balanced', random_state=42)
                params = {'clf__max_depth': [10, 20]}
            else:
                clf = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
                params = {'clf__C': [0.1, 1.0]}

            pipeline = Pipeline([('pre', preprocessor), ('clf', clf)])
            search = RandomizedSearchCV(pipeline, params, n_iter=2, cv=2, scoring='recall', verbose=1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            # Gender Equity Audit
            y_pred = best_model.predict(X_test)
            res = df_test_final.copy(); res['pred'] = y_pred
            print(f"\n⚖️ Gender Drilldown ({model_type}):")
            print(res.groupby('gender').agg(Actual=('churn', 'mean'), Predicted=('pred', 'mean')))

            # Metrics
            recall = recall_score(y_test, y_pred)
            mlflow.log_metric("test_recall", recall)

            # SHAP (Enhanced)
            try:
                sample = X_test.sample(min(100, len(X_test)))
                X_mapped = pd.DataFrame(best_model.named_steps['pre'].transform(sample), columns=best_model.named_steps['pre'].get_feature_names_out())
                
                if model_type == "logistic_regression":
                    explainer = shap.LinearExplainer(best_model.named_steps['clf'], X_mapped)
                else:
                    explainer = shap.Explainer(best_model.named_steps['clf'], X_mapped)
                    
                shap_values = explainer(X_mapped)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_mapped, show=False, max_display=20)
                plt.savefig(f"shap_{model_type}.png")
                mlflow.log_artifact(f"shap_{model_type}.png")
                plt.close()
            except Exception as e:
                print(f"⚠️ SHAP Skip: {e}")

            # Register
            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.sklearn.log_model(best_model, "model", registered_model_name=f"CustomerChurnModel_{model_type}", signature=signature)
            print(f"✅ {model_type.upper()} Model Registered.")

if __name__ == "__main__":
    train_and_register()
