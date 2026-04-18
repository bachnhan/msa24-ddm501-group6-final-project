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
from sklearn.metrics import recall_score, classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from mlflow.models import infer_signature
from dotenv import load_dotenv

# Load Local Secrets
load_dotenv(".secret")

def clean_telco_data(df):
    """Specific cleaning for the Telco Customer Churn dataset."""
    df.columns = [col.lower() for col in df.columns]
    # Remove ID
    df = df.drop(columns=[c for c in ['customerid'] if c in df.columns])
    # Handle numeric conversion for TotalCharges (has empty spaces)
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    # Convert Target to numeric
    if 'churn' in df.columns:
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
    return df.dropna()

def train_and_register():
    # 1. Setup & Auth
    dagshub.init(repo_owner="nhannhb92", repo_name="msa24-ddm501-group6-final-project", mlflow=True)
    mlflow.sklearn.autolog(log_models=False)
    
    # 2. Data Loading (Local or Download)
    print("📂 Loading TELCO MASTER DATA (V6.0)...")
    try:
        # Assuming the CSV is in the workspace after Kaggle download
        path = "telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if not os.path.exists(path):
             print("⚠️ Telco CSV not found locally. Please run the notebook first.")
             return
        df_raw = clean_telco_data(pd.read_csv(path))
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    df_train, df_test = train_test_split(df_raw, test_size=0.2, random_state=42, stratify=df_raw['churn'])
    
    print(f"✅ Data Profile: Train({len(df_train)}), Test({len(df_test)})")
    print(f"📊 Global Churn Rate: {df_raw['churn'].mean():.2%}")

    X_train, y_train = df_train.drop(columns=['churn']), df_train['churn']
    X_test, y_test = df_test.drop(columns=['churn']), df_test['churn']

    # 3. Preprocessing
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])

    # 4. Training (Optimized for Telco)
    for m_type in ["xgboost", "logistic_regression"]:
        mlflow.set_experiment("Churn_Telco_Production")
        with mlflow.start_run(run_name=f"SCRIPT_TELCO_{m_type.upper()}"):
            print(f"\n🚀 Training {m_type.upper()}...")
            
            if m_type == "xgboost":
                pos_w = (y_train == 0).sum() / (y_train == 1).sum()
                clf = XGBClassifier(scale_pos_weight=pos_w, n_estimators=100, max_depth=5, eval_metric='logloss')
                m_params = {'clf__max_depth': [3, 5, 7]}
            else:
                clf = LogisticRegression(max_iter=2000, class_weight='balanced')
                m_params = {'clf__C': [0.1, 1.0, 10.0]}

            pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
            search = RandomizedSearchCV(pipe, m_params, n_iter=3, cv=3, scoring='roc_auc')
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            # --- Analysis ---
            y_pred = best_model.predict(X_test)
            print(f"🏆 Results for {m_type.upper()}:")
            print(classification_report(y_test, y_pred))
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

            # Logs & Registry
            mlflow.sklearn.log_model(best_model, "model", registered_model_name=f"CustomerChurnModel_{m_type}")
            print(f"✅ Registered: CustomerChurnModel_{m_type}")

if __name__ == "__main__":
    train_and_register()
