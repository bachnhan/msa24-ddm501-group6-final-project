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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models import infer_signature
import shap
import logging

# 1. SETUP & LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(override=True)
for secret_file in [".secret", "/etc/secrets/.env", "/etc/secrets/.secret"]:
    if os.path.exists(secret_file):
        load_dotenv(secret_file, override=True)

# Activate MLflow autologging for deep transparency
mlflow.sklearn.autolog(log_models=False)

# ---------------------------------------------------------
# 2. GLOBAL PREPARATION
# ---------------------------------------------------------
def clean_df(df):
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    df = df.drop(columns=[c for c in ['customerid', 'last_interaction'] if c in df.columns])
    if 'age' in df.columns:
        df = df[(df['age'] >= 18) & (df['age'] <= 120)]
    for col in [c for c in ['total_spend', 'totalcharges'] if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['churn'])

def prepare_data():
    repo_root = Path(__file__).parent.parent
    train_path = repo_root / "data" / "customer_churn_dataset-training-master.csv"
    test_path = repo_root / "data" / "customer_churn_dataset-testing-master.csv"
    
    df_raw_tr = clean_df(pd.read_csv(train_path))
    df_test = clean_df(pd.read_csv(test_path))
    df_tr, df_va = train_test_split(df_raw_tr, test_size=0.2, random_state=42, stratify=df_raw_tr['churn'])
    
    X_tr = df_tr.drop(columns=['churn']); y_tr = df_tr['churn']
    X_va = df_va.drop(columns=['churn']); y_va = df_va['churn']
    X_te = df_test.drop(columns=['churn']); y_te = df_test['churn']
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), X_tr.select_dtypes(include=[np.number]).columns.tolist()),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), X_tr.select_dtypes(include=['object']).columns.tolist())
    ])
    
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    return X_tr, X_va, X_te, y_tr, y_va, y_te, preprocessor, pos_weight, df_test

# ---------------------------------------------------------
# 3. FULL AUDIT RUNNER (RECALL PRIORITY)
# ---------------------------------------------------------
def run_audit_experiment(X_tr, X_va, X_te, y_tr, y_va, y_te, preprocessor, pos_weight, df_te, model_type):
    reg_name = f"CustomerChurnModel_{model_type}"
    with mlflow.start_run(run_name=f"SCRIPT_AUDIT_{model_type.upper()}"):
        logger.info(f"🔥 Training {model_type.upper()} with Full Audit...")
        
        if model_type == "xgboost":
            clf = XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42)
            params = {'clf__max_depth': [3, 5]}
        elif model_type == "random_forest":
            clf = RandomForestClassifier(class_weight='balanced', random_state=42)
            params = {'clf__max_depth': [10, 20]}
        else:
            clf = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
            params = {'clf__C': [0.1, 1.0]}

        pipeline = Pipeline([('pre', preprocessor), ('clf', clf)])
        search = RandomizedSearchCV(pipeline, params, n_iter=1, cv=2, scoring='recall', verbose=1)
        search.fit(X_tr, y_tr)
        best_model = search.best_estimator_
        
        y_pred = best_model.predict(X_te)
        y_probs = best_model.predict_proba(X_te)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred
        
        # [Visual 1] ROC
        fpr, tpr, _ = roc_curve(y_te, y_probs)
        plt.figure(); plt.plot(fpr, tpr, label=f'AUC={auc(fpr,tpr):.2f}'); plt.title(f"ROC: {model_type}"); plt.savefig(f"roc_{model_type}.png"); mlflow.log_artifact(f"roc_{model_type}.png"); plt.close()
        
        # [Visual 2] Fairness Case
        if 'gender' in df_te.columns:
            df_audit = df_te.copy(); df_audit['err'] = np.abs(y_te - y_pred)
            gap = df_audit.groupby('gender')['err'].mean().diff().abs().iloc[-1]
            mlflow.log_metric("gender_bias_gap", gap)
            
        # [Visual 3] Explainability (SHAP)
        try:
            sample = X_te.sample(min(100, len(X_te)))
            feature_names = best_model.named_steps['pre'].get_feature_names_out()
            X_mapped = pd.DataFrame(best_model.named_steps['pre'].transform(sample), columns=feature_names)
            explainer = shap.Explainer(best_model.named_steps['clf'])
            shap_values = explainer(X_mapped)
            plt.figure(); shap.summary_plot(shap_values, X_mapped, show=False)
            plt.savefig(f"shap_{model_type}.png"); mlflow.log_artifact(f"shap_{model_type}.png"); plt.close()
        except: pass

        mlflow.log_metrics({"test_recall": recall_score(y_te, y_pred), "test_f1": f1_score(y_te, y_pred)})
        signature = infer_signature(X_tr, best_model.predict(X_tr))
        mlflow.sklearn.log_model(best_model, "model", registered_model_name=reg_name, signature=signature)
        logger.info(f"✅ Completed & Registered {model_type.upper()}")

def main():
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri: mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("Churn_Final_Script_Audit")
    
    X_tr, X_va, X_te, y_tr, y_va, y_te, prep, weight, df_te = prepare_data()
    for m in ["xgboost", "logistic_regression", "random_forest"]:
        try: run_audit_experiment(X_tr, X_va, X_te, y_tr, y_va, y_te, prep, weight, df_te, m)
        except Exception as e: logger.error(f"Error training {m}: {e}")

if __name__ == "__main__":
    main()
