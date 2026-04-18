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
from sklearn.metrics import recall_score, classification_report
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
from dotenv import load_dotenv

# Load Local Secrets
load_dotenv(".secret")

def clean_dataset(df):
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    df = df.drop(columns=[c for c in ['customerid', 'last_interaction'] if c in df.columns])
    if 'age' in df.columns:
        df = df[(df['age'] >= 18) & (df['age'] <= 120)]
    for col in [c for c in ['total_spend', 'totalcharges'] if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['churn'])

def train_and_register():
    dagshub.init(repo_owner=\"nhannhb92\", repo_name=\"msa24-ddm501-group6-final-project\", mlflow=True)
    mlflow.sklearn.autolog(log_models=False)
    
    try:
        df_raw = clean_dataset(pd.read_csv(\"customer-churn-dataset/customer_churn_dataset-training-master.csv\"))
        df_test_final = clean_dataset(pd.read_csv(\"customer-churn-dataset/customer_churn_dataset-testing-master.csv\"))
    except:
        return

    df_train, df_val = train_test_split(df_raw, test_size=0.15, random_state=42, stratify=df_raw['churn'])
    
    X_train, y_train = df_train.drop(columns=['churn']), df_train['churn']
    X_val, y_val = df_val.drop(columns=['churn']), df_val['churn']
    X_test, y_test = df_test_final.drop(columns=['churn']), df_test_final['churn']

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), X_train.select_dtypes(include=[np.number]).columns.tolist()),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), X_train.select_dtypes(include=['object']).columns.tolist())
    ])

    preprocessor.fit(X_train)
    X_val_t = preprocessor.transform(X_val)

    for m_type in [\"xgboost\", \"random_forest\", \"logistic_regression\"]:
        mlflow.set_experiment(\"Churn_Final_Audit\")
        with mlflow.start_run(run_name=f\"SCRIPT_{m_type.upper()}\"):
            print(f\"\\n🚀 Training {m_type.upper()}...\")
            
            if m_type == \"xgboost\":
                clf = XGBClassifier(scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(), early_stopping_rounds=10)
                f_params = {\"clf__eval_set\": [(X_val_t, y_val)], \"clf__verbose\": False}
                m_params = {'clf__max_depth': [3, 5]}
            elif m_type == \"random_forest\":
                clf = RandomForestClassifier(class_weight='balanced', random_state=42)
                f_params = {}; m_params = {'clf__max_depth': [10, 20]}
            else:
                clf = LogisticRegression(class_weight='balanced', max_iter=2000)
                f_params = {}; m_params = {'clf__C': [0.1, 1.0]}

            pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
            search = RandomizedSearchCV(pipe, m_params, n_iter=2, cv=2, scoring='recall', verbose=1)
            search.fit(X_train, y_train, **f_params)
            best_model = search.best_estimator_

            # SHAP (Improved Logic)
            try:
                sample = X_test.sample(min(200, len(X_test)))
                X_t = pd.DataFrame(best_model.named_steps['pre'].transform(sample), columns=best_model.named_steps['pre'].get_feature_names_out())
                explainer = shap.Explainer(best_model.named_steps['clf'], X_t) if m_type == \"logistic_regression\" else shap.Explainer(best_model.named_steps['clf'])
                plt.figure(figsize=(10, 8)); shap.summary_plot(explainer(X_t), X_t, show=False, max_display=20)
                plt.savefig(f\"shap_{m_type}.png\"); mlflow.log_artifact(f\"shap_{m_type}.png\"); plt.close()
            except: pass

            mlflow.sklearn.log_model(best_model, \"model\", registered_model_name=f\"CustomerChurnModel_{m_type}\")

if __name__ == \"__main__\":
    train_and_register()
