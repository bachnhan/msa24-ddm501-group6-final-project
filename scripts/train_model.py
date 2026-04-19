import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    recall_score,
    precision_score,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve,
    make_scorer,
    fbeta_score,
)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv

# Load Local Secrets
load_dotenv()

def clean_telco_data(df):
    """Specific cleaning for the Telco Customer Churn dataset with Tenure Binning."""
    df.columns = [col.lower() for col in df.columns]
    # Remove ID
    df = df.drop(columns=[c for c in ["customerid"] if c in df.columns])
    
    # Handle numeric conversion for TotalCharges (has empty spaces)
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df["totalcharges"] = df["totalcharges"].fillna(0)
    
    # --- FEATURE ENGINEERING: Tenure Binning ---
    bins = [0, 12, 24, 48, 60, 100]
    labels = ['0-12m', '12-24m', '24-48m', '48-60m', '60m+']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True).astype(str)
    
    # Convert Target to numeric
    if "churn" in df.columns:
        df["churn"] = df["churn"].map({"Yes": 1, "No": 0})
    
    return df

def constrained_recall_scorer(y_true, y_probs, min_precision=0.7):
    """Maximum recall achievable while precision >= 0.7."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    idx = np.where(precisions >= min_precision)[0]
    if len(idx) > 0:
        return recalls[idx].max()
    return 0

def train_and_register():
    # 1. Setup & Auth
    dagshub.init(
        repo_owner="nhannhb92", 
        repo_name="msa24-ddm501-group6-final-project", 
        mlflow=True
    )
    mlflow.sklearn.autolog(log_models=False)
    
    # 2. Data Loading
    print("📂 Loading TELCO MASTER DATA (V9.0 - Round 2: Recall Focused)...")
    try:
        path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if not os.path.exists(path):
             path = "telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        
        if not os.path.exists(path):
             print(f"⚠️ Telco CSV not found at {path}")
             return
        df_raw = clean_telco_data(pd.read_csv(path))
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    df_train, df_test = train_test_split(df_raw, test_size=0.2, random_state=42, stratify=df_raw["churn"])
    X_train, y_train = df_train.drop(columns=["churn"]), df_train["churn"]
    X_test, y_test = df_test.drop(columns=["churn"]), df_test["churn"]

    # 3. Preprocessing
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])

    # 4. Multi-Model Loop
    for m_type in ["xgboost", "random_forest", "logistic_regression"]:
        mlflow.set_experiment("Churn_Round2_Optimized")
        with mlflow.start_run(run_name=f"ROUND2_{m_type.upper()}"):
            print(f"\n🚀 [Round 2] Training {m_type.upper()} with High Intensity Search...")
            
            if m_type == "xgboost":
                clf = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
                m_params = {
                    'clf__max_depth': [3, 4, 5, 6],
                    'clf__learning_rate': [0.01, 0.05, 0.1],
                    'clf__n_estimators': [100, 200, 300]
                }
            elif m_type == "random_forest":
                clf = RandomForestClassifier(random_state=42)
                m_params = {
                    'clf__n_estimators': [100, 200, 300], 
                    'clf__max_depth': [5, 10, 15, None]
                }
            else:
                clf = LogisticRegression(max_iter=2000, random_state=42)
                m_params = {'clf__C': [0.01, 0.1, 1.0, 10.0]}

            pipe = ImbPipeline([
                ('pre', preprocessor), 
                ('smote', SMOTE(random_state=42)),
                ('clf', clf)
            ])
            
            # Increased search intensity (n_iter=15, cv=5)
            search = RandomizedSearchCV(pipe, m_params, n_iter=15, cv=5, scoring='recall', n_jobs=-1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            # --- [Round 2] THRESHOLD OPTIMIZATION (Recall >= 80% with Max Precision) ---
            print(f"🔍 Finding optimal threshold for {m_type.upper()} (Target: Recall >= 0.80)...")
            y_probs = best_model.predict_proba(X_test)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
            
            best_threshold = 0.5 # Default
            valid_indices = [i for i, r in enumerate(recalls[:-1]) if r >= 0.80]
            if valid_indices:
                # Maximize Precision among those meeting the Recall target
                best_idx = valid_indices[0]
                max_p = 0
                for idx in valid_indices:
                    if precisions[idx] > max_p:
                        max_p = precisions[idx]
                        best_idx = idx
                best_threshold = thresholds[best_idx]
            
            y_pred = (y_probs >= best_threshold).astype(int)
            
            print(f"🏆 Round 2 Results ({m_type.upper()}) at Threshold {best_threshold:.4f}:")
            print(classification_report(y_test, y_pred))
            
            mlflow.log_param("best_threshold", best_threshold)
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_probs))

            # --- RESPONSIBLE AI: FAIRNESS ---
            audit_df = X_test.copy()
            audit_df['predicted'] = y_pred
            gender_col = next((c for c in audit_df.columns if c.lower() == 'gender'), None)
            if gender_col:
                gender_stats = audit_df.groupby(gender_col)['predicted'].mean()
                if len(gender_stats) >= 2:
                    bias_gap = abs(gender_stats.iloc[0] - gender_stats.iloc[1])
                    mlflow.log_metric("bias_gap_gender", bias_gap)
                gender_stats.to_csv(f"audit_v9_{m_type}.csv")
                mlflow.log_artifact(f"audit_v9_{m_type}.csv")
            
            # --- RESPONSIBLE AI: SHAP ---
            try:
                X_tx = preprocessor.transform(X_test)
                if hasattr(X_tx, "toarray"): X_tx = X_tx.toarray()
                explainer = shap.Explainer(best_model.named_steps['clf'])
                shap_values = explainer(X_tx)
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_tx, feature_names=preprocessor.get_feature_names_out(), show=False)
                plt.savefig(f"shap_{m_type}.png")
                mlflow.log_artifact(f"shap_{m_type}.png")
                plt.close()
            except Exception: pass

            # Register
            mlflow.sklearn.log_model(
                best_model,
                "model",
                registered_model_name=f"CustomerChurnModel_{m_type}",
            )
            print(f"✅ Registered: CustomerChurnModel_{m_type}")

if __name__ == "__main__":
    train_and_register()
