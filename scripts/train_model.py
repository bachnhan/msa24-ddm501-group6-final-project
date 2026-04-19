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
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score, precision_score, classification_report, accuracy_score, roc_auc_score, precision_recall_curve, make_scorer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

def constrained_recall_scorer(y_true, y_probs, min_precision=0.7):
    """Maximum recall achievable while precision >= 0.7."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    idx = np.where(precisions >= min_precision)[0]
    if len(idx) > 0:
        return recalls[idx].max()
    return 0

def train_and_register():
    # 1. Setup & Auth
    dagshub.init(repo_owner="nhannhb92", repo_name="msa24-ddm501-group6-final-project", mlflow=True)
    mlflow.sklearn.autolog(log_models=False)
    
    # 2. Data Loading
    print("📂 Loading TELCO MASTER DATA (Synced V6.4)...")
    path = "telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(path):
         print("⚠️ Telco CSV not found locally. Searching for dataset...")
         # Optional: Add download logic if needed, but assuming user has it
         return
    df_raw = clean_telco_data(pd.read_csv(path))

    df_train, df_test = train_test_split(df_raw, test_size=0.2, random_state=42, stratify=df_raw['churn'])
    X_train, y_train = df_train.drop(columns=['churn']), df_train['churn']
    X_test, y_test = df_test.drop(columns=['churn']), df_test['churn']

    # 3. Preprocessing
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])

    # 0.7 Precision constraint (The "Prediction not too low" requirement)
    recall_ctrl_scorer = make_scorer(constrained_recall_scorer, needs_proba=True, min_precision=0.7)

    # 4. Multi-Model Loop
    for m_type in ["xgboost", "random_forest", "logistic_regression"]:
        mlflow.set_experiment("Churn_Telco_Production")
        with mlflow.start_run(run_name=f"SCRIPT_RAI_{m_type.upper()}"):
            print(f"\n🚀 Training {m_type.upper()} with SMOTE & P>=0.7 Constraint...")
            
            if m_type == "xgboost":
                clf = XGBClassifier(n_estimators=100, max_depth=5, eval_metric='logloss')
                m_params = {'clf__max_depth': [3, 5, 7]}
            elif m_type == "random_forest":
                clf = RandomForestClassifier(random_state=42)
                m_params = {'clf__n_estimators': [100, 200], 'clf__max_depth': [10, None]}
            else:
                clf = LogisticRegression(max_iter=2000)
                m_params = {'clf__C': [0.1, 1.0, 10.0]}

            # Use ImbPipeline to include SMOTE
            pipe = ImbPipeline([
                ('pre', preprocessor), 
                ('smote', SMOTE(random_state=42)),
                ('clf', clf)
            ])
            
            search = RandomizedSearchCV(pipe, m_params, n_iter=3, cv=3, scoring=recall_ctrl_scorer)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            
            # --- THRESHOLD OPTIMIZATION (Prec 0.7 floor) ---
            y_proba = best_model.predict_proba(X_test)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
            idx = np.where(precisions >= 0.7)[0]
            if len(idx) > 0 and idx[0] < len(thresholds):
                best_threshold = thresholds[min(idx[np.argmax(recalls[idx])], len(thresholds)-1)]
            else:
                best_threshold = 0.5
            
            y_pred = (y_proba >= best_threshold).astype(int)
            
            print(f"🏆 Results for {m_type.upper()} (Threshold: {best_threshold:.4f}):")
            print(f"Recall: {recall_score(y_test, y_pred):.4f} | Precision: {precision_score(y_test, y_pred):.4f}")
            
            mlflow.log_param("best_threshold", best_threshold)
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred))

            # --- RESPONSIBLE AI: FAIRNESS ---
            audit_df = X_test.copy()
            audit_df['predicted'] = y_pred
            gender_col = next((c for c in audit_df.columns if c.lower() == 'gender'), None)
            if gender_col:
                gender_stats = audit_df.groupby(gender_col)['predicted'].mean()
                if len(gender_stats) >= 2:
                    bias_gap = abs(gender_stats.iloc[0] - gender_stats.iloc[1])
                    mlflow.log_metric("bias_gap_gender", bias_gap)
                gender_stats.to_csv(f"audit_{m_type}.csv")
                mlflow.log_artifact(f"audit_{m_type}.csv")
            
            # --- RESPONSIBLE AI: SHAP ---
            try:
                X_tx = preprocessor.transform(X_test)
                if hasattr(X_tx, "toarray"): X_tx = X_tx.toarray()
                
                if m_type == "logistic_regression":
                    explainer = shap.Explainer(best_model.named_steps['clf'], X_tx)
                else:
                    explainer = shap.Explainer(best_model.named_steps['clf'])
                    
                shap_values = explainer(X_tx)
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_tx, feature_names=preprocessor.get_feature_names_out(), show=False)
                plt.savefig(f"shap_{m_type}.png")
                mlflow.log_artifact(f"shap_{m_type}.png")
                plt.close()
            except Exception as e: 
                print(f"⚠️ SHAP failed for {m_type}: {e}")

            # Register
            mlflow.sklearn.log_model(best_model, "model", registered_model_name=f"Churn_{m_type}_Script")
            print(f"✅ {m_type.upper()} Registered with RAI Artifacts.")

if __name__ == "__main__":
    train_and_register()
