import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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
    confusion_matrix, 
    roc_auc_score,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve
)
from dotenv import load_dotenv

# Load Local Secrets
load_dotenv()
if os.path.exists(".secret"):
    load_dotenv(".secret")

# Explicitly set MLflow tracking URI from environment if available
if "MLFLOW_TRACKING_URI" in os.environ:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

def clean_telco_data(df):
    """Clean and preprocess the Telco Customer Churn dataset."""
    df.columns = [col.lower() for col in df.columns]
    
    # Remove customerID as it's not a feature
    if 'customerid' in df.columns:
        df = df.drop(columns=['customerid'])
    
    # Handle TotalCharges where spaces might exist for new customers (0 tenure)
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    df['totalcharges'] = df['totalcharges'].fillna(0)
    
    # Convert Target 'churn' to numeric
    if 'churn' in df.columns:
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
    
    return df

def train_xgboost_with_recall_focus():
    # 1. Initialize Experiment Tracking (DISABLED as per user request)
    print("🚀 Starting Advanced XGBoost training (Score: F2, Threshold: Optimized)...")

    # 2. Data Loading
    print("📂 Loading dataset for Recall optimization...")
    data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        return

    df_raw = pd.read_csv(data_path)
    df = clean_telco_data(df_raw)
    
    # Define features and target
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Preprocessing Pipeline
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('sc', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

    # 4. XGBoost Hyperparameter Tuning focused on F2 (Better Recall/Precision balance)
    f2_scorer = make_scorer(fbeta_score, beta=2)

    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [3, 4, 5, 6],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__subsample': [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0]
    }

    xgb_pipe = ImbPipeline([
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    print("🚀 Running GridSearchCV for Recall optimization...")
    # Training locally without MLflow
    grid_search = GridSearchCV(
        xgb_pipe, 
        param_grid, 
        cv=3, 
        scoring='recall', 
        n_jobs=-1, 
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # 5. Evaluation with 0.5 Prediction Threshold
    threshold = 0.5
    y_probs = best_model.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_probs >= threshold).astype(int)
    
    # Metrics calculation
    recall = recall_score(y_test, y_pred_custom)
    precision = precision_score(y_test, y_pred_custom)
    f1 = f1_score(y_test, y_pred_custom)
    auc = roc_auc_score(y_test, y_probs)
    
    print("\n🏆 Best Parameters:", grid_search.best_params_)
    print(f"📊 Metrics at Threshold {threshold}:")
    print(f"   Recall:    {recall*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}%")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC AUC:   {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_custom))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred_custom)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Threshold: {threshold})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("confusion_matrix_recall.png")
    print("✅ Confusion matrix saved as 'confusion_matrix_recall.png'.")
    plt.close()

if __name__ == "__main__":
    train_xgboost_with_recall_focus()
