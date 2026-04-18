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
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.models import infer_signature
import shap

# 1. SETUP
load_dotenv(override=True)
for secret_file in [".secret", "/etc/secrets/.env", "/etc/secrets/.secret"]:
    if os.path.exists(secret_file):
        load_dotenv(secret_file, override=True)

# ---------------------------------------------------------
# 2. GLOBAL PREPARATION
# ---------------------------------------------------------
def clean_df(df):
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    # Standard Anti-Leakage
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
    return X_tr, X_va, X_te, y_tr, y_va, y_te, preprocessor, pos_weight

# ---------------------------------------------------------
# 3. EXPERIMENT RUNNER (RECALL PRIORITY)
# ---------------------------------------------------------
def run_experiment(X_tr, X_va, X_te, y_tr, y_va, y_te, preprocessor, pos_weight, model_type):
    reg_name = f"CustomerChurnModel_{model_type}"
    with mlflow.start_run(run_name=f"FINAL_RECALL_{model_type.upper()}"):
        print(f"🚀 Training {model_type.upper()}...")
        
        if model_type == "xgboost":
            clf = XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42)
            params = {'clf__n_estimators': [100, 200]}
        elif model_type == "random_forest":
            clf = RandomForestClassifier(random_state=42)
            params = {'clf__n_estimators': [100, 200]}
        else:
            clf = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
            params = {'clf__C': [0.1, 1.0]}

        pipeline = Pipeline([('pre', preprocessor), ('clf', clf)])
        search = RandomizedSearchCV(pipeline, params, n_iter=2, cv=3, scoring='recall', n_jobs=-1)
        search.fit(X_tr, y_tr)
        best_model = search.best_estimator_
        
        # Metrics & Plots
        y_pred = best_model.predict(X_te)
        y_probs = best_model.predict_proba(X_te)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred
        
        # [Visual] ROC
        fpr, tpr, _ = roc_curve(y_te, y_probs)
        plt.figure(); plt.plot(fpr, tpr, label=f'AUC={auc(fpr,tpr):.2f}'); plt.savefig(f"roc_{model_type}.png"); mlflow.log_artifact(f"roc_{model_type}.png"); plt.close()
        
        mlflow.log_metrics({"recall": recall_score(y_te, y_pred), "f1_score": f1_score(y_te, y_pred)})
        
        signature = infer_signature(X_tr, best_model.predict(X_tr))
        mlflow.sklearn.log_model(best_model, "model", registered_model_name=reg_name, signature=signature)
        print(f"      ✅ Registered {reg_name}")

def main():
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri: mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("Churn_Final_Recall_Optimization")
    
    X_tr, X_va, X_te, y_tr, y_va, y_te, prep, weight = prepare_data()
    for m in ["xgboost", "logistic_regression", "random_forest"]:
        try: run_experiment(X_tr, X_va, X_te, y_tr, y_va, y_te, prep, weight, m)
        except Exception as e: print(f"❌ Error training {m}: {e}")

if __name__ == "__main__":
    main()
