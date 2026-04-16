# TELCO CUSTOMER CHURN PREDICTION SYSTEM

## Problem Definition & Requirements

### Section B — User Requirements, Use Cases & Scope Constraints

**Course:** DDM501 - AI in Production: From Models to Systems
**Dataset:** IBM Telco Customer Churn
**Prepared by:** Lê Huỳnh Trang

---

## 1. STAKEHOLDER IDENTIFICATION

| Stakeholder                    | Type           | Interaction with System                                                  |
| ------------------------------ | -------------- | ------------------------------------------------------------------------ |
| **Retention Manager**    | Primary User   | Receives daily high-risk customer lists; triggers intervention campaigns |
| **Call Center Agent**    | Primary User   | Sees real-time churn risk score during customer calls via CRM widget     |
| **Business Analyst**     | Secondary User | Queries aggregate churn reports; analyzes feature importance trends      |
| **ML Engineer**          | Operator       | Monitors model health; triggers retraining; reviews drift alerts         |
| **IT / DevOps**          | Operator       | Maintains infrastructure; manages Docker services and API uptime         |
| **Executive / VP Sales** | Observer       | Reviews weekly business dashboard (revenue at risk, retention rate)      |

---

## 2. FUNCTIONAL REQUIREMENTS

> Priority is assigned using **MoSCoW**: **M** = Must Have | **S** = Should Have | **C** = Could Have | **W** = Won't Have (this iteration)

### 2.1. Data Pipeline

| ID     | Requirement                                                                                                                | Priority    |
| ------ | -------------------------------------------------------------------------------------------------------------------------- | ----------- |
| FR-D01 | System**must** ingest the IBM Telco Customer Churn CSV dataset                                                       | **M** |
| FR-D02 | System**must** detect and handle whitespace values in `TotalCharges` (known data quality issue)                    | **M** |
| FR-D03 | System**must** perform One-Hot Encoding on all categorical features (Contract, InternetService, PaymentMethod, etc.) | **M** |
| FR-D04 | System**must** normalize numerical features (Tenure, MonthlyCharges, TotalCharges) using StandardScaler              | **M** |
| FR-D05 | System**should** apply SMOTE or class-weight adjustment to handle the ~26.5% class imbalance                         | **S** |
| FR-D06 | System**should** version processed datasets using MLflow artifact logging                                            | **S** |
| FR-D07 | System **could** automatically detect schema drift in incoming data (e.g., missing columns)                         | **C** |

### 2.2. Model Training & Experimentation

| ID     | Requirement                                                                                                | Priority    |
| ------ | ---------------------------------------------------------------------------------------------------------- | ----------- |
| FR-M01 | System**must** train at least one classification model (baseline: Logistic Regression)               | **M** |
| FR-M02 | System**must** train XGBoost or LightGBM as the primary candidate model                              | **M** |
| FR-M03 | System**must** evaluate models using ROC-AUC, F1-Score, Precision, and Recall on a held-out test set | **M** |
| FR-M04 | System**must** log all experiments (parameters, metrics, artifacts) to MLflow Tracking Server        | **M** |
| FR-M05 | System**must** perform cross-validation (k=5) during model selection                                 | **M** |
| FR-M06 | System**should** perform hyperparameter tuning (GridSearchCV or Optuna) for the best candidate model | **S** |
| FR-M07 | System**should** generate SHAP summary plots and per-prediction feature attributions                 | **S** |
| FR-M08 | System**could** train a deep learning model (TabNet) as an experimental challenger                   | **C** |
| FR-M09 | System**won't** perform real-time online learning in this iteration                                  | **W** |

### 2.3. Prediction API

| ID     | Requirement                                                                                                | Priority    |
| ------ | ---------------------------------------------------------------------------------------------------------- | ----------- |
| FR-A01 | System**must** expose a REST API endpoint `POST /predict` accepting a JSON customer feature vector | **M** |
| FR-A02 | API**must** return a churn probability score (0.0–1.0) and a binary prediction                      | **M** |
| FR-A03 | API**must** return a `reason_codes` list: top 3 SHAP features driving the prediction               | **M** |
| FR-A04 | API**must** expose a health check endpoint `GET /health`                                           | **M** |
| FR-A05 | API**should** expose a batch endpoint `POST /predict/batch` accepting a list of customers          | **S** |
| FR-A06 | API**should** include API versioning (`/api/v1/predict`)                                           | **S** |
| FR-A07 | API**should** include input validation and return descriptive 422/400 error messages                 | **S** |
| FR-A08 | API**could** expose a `GET /model-info` endpoint returning current model version and metrics       | **C** |

### 2.4. Monitoring & Alerting

| ID     | Requirement                                                                                                         | Priority    |
| ------ | ------------------------------------------------------------------------------------------------------------------- | ----------- |
| FR-O01 | System**must** expose Prometheus metrics: request count, latency histogram, error rate                        | **M** |
| FR-O02 | System**must** display a Grafana dashboard with API health, prediction distribution, and model performance    | **M** |
| FR-O03 | System**should** monitor data drift on `MonthlyCharges`, `Tenure`, and `Contract` feature distributions | **S** |
| FR-O04 | System**should** fire an alert if prediction error rate exceeds 5% over a 1-hour window                       | **S** |
| FR-O05 | System**could** track population stability index (PSI) for feature drift detection                            | **C** |

### 2.5. CI/CD & Testing

| ID     | Requirement                                                                                               | Priority    |
| ------ | --------------------------------------------------------------------------------------------------------- | ----------- |
| FR-C01 | System**must** have a GitHub Actions workflow that runs unit tests on every pull request            | **M** |
| FR-C02 | System**must** have unit tests covering data preprocessing and model scoring functions              | **M** |
| FR-C03 | System**must** have integration tests for all API endpoints                                         | **M** |
| FR-C04 | System**should** have data quality tests validating schema, null checks, and value ranges           | **S** |
| FR-C05 | System**should** have a model validation gate: new model must outperform baseline before deployment | **S** |
| FR-C06 | System**should** auto-build and push Docker image on merge to `main` branch                       | **S** |

---

## 3. NON-FUNCTIONAL REQUIREMENTS

| ID     | Category                  | Requirement                                                                                      | Target                    |
| ------ | ------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------- |
| NFR-01 | **Performance**     | API p95 inference latency                                                                        | < 150ms                   |
| NFR-02 | **Performance**     | Batch scoring throughput (7,000 records)                                                         | < 30 seconds              |
| NFR-03 | **Availability**    | API uptime                                                                                       | ≥ 99.5%                  |
| NFR-04 | **Scalability**     | API must handle concurrent requests without degradation                                          | ≥ 10 concurrent requests |
| NFR-05 | **Reliability**     | All Docker services must have health checks and auto-restart policies                            | Required                  |
| NFR-06 | **Security**        | API must not expose raw training data via any endpoint                                           | Required                  |
| NFR-07 | **Security**        | No customer PII (e.g., real names, phone numbers) stored in logs                                 | Required                  |
| NFR-08 | **Maintainability** | All functions must have docstrings; type hints required for public methods                       | Required                  |
| NFR-09 | **Maintainability** | Test coverage must be ≥ 80% on core modules                                                     | ≥ 80%                    |
| NFR-10 | **Portability**     | System must run fully via `docker compose up` with no manual setup steps                       | Required                  |
| NFR-11 | **Explainability**  | Every prediction must return human-readable reason codes                                         | Required                  |
| NFR-12 | **Fairness**        | Model must not show demographic parity difference > 0.05 across gender and senior citizen groups | ≤ 0.05                   |

---

## 4. USE CASES

### UC-01: Real-Time Churn Scoring During Customer Call

| Field                      | Detail                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Use Case ID**      | UC-01                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Name**             | Real-Time Churn Scoring                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Actor**            | Call Center Agent (via CRM Widget)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Trigger**          | Customer initiates an inbound support or billing call                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Precondition**     | Customer exists in the system; API service is running                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Main Flow**        | 1. CRM system identifies the calling customer by phone number.`<br>` 2. CRM sends `POST /api/v1/predict` with the customer's current feature snapshot. `<br>` 3. API returns `churn_probability: 0.82`, `prediction: "Churn"`, and `reason_codes: ["Month-to-month contract", "High monthly charges ($89/mo)", "No Tech Support"]`. `<br>` 4. CRM displays a **🔴 High Risk** badge on the agent's screen with the reason codes. `<br>` 5. Agent offers a targeted intervention (e.g., convert to 1-year contract with 15% discount). |
| **Alternative Flow** | A2: If API returns latency > 2s, CRM falls back to a "Last Known Score" cached value.`<br>` A3: If customer is not found, CRM displays "No churn data available."                                                                                                                                                                                                                                                                                                                                                                                         |
| **Postcondition**    | Agent's intervention is logged in CRM; customer's next prediction cycle accounts for updated contract status.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Business Value**   | Prevents churn at the highest-leverage moment: when the customer is already engaged on the phone.                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

---

### UC-02: Automated Weekly Batch Risk Report

| Field                      | Detail                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Use Case ID**      | UC-02                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Name**             | Weekly Batch Churn Risk Report                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Actor**            | Business Analyst / Retention Manager (scheduled job)                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Trigger**          | Monday 7:00 AM automated cron job                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Precondition**     | Updated customer data has been loaded into the system                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Main Flow**        | 1. Cron scheduler triggers `POST /api/v1/predict/batch` with all active customers. `<br>` 2. API scores all customers and returns a ranked list by `churn_probability`. `<br>` 3. System segments customers into three risk tiers: **High (≥ 0.70)**, **Medium (0.40–0.69)**, **Low (< 0.40)**. `<br>` 4. Report is exported as CSV and pushed to the Retention Manager's dashboard. `<br>` 5. High-risk list (~500 customers) is automatically queued for outbound call campaigns. |
| **Alternative Flow** | A2: If batch job fails, alert is sent to ML Engineer via Prometheus/Alertmanager.                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| **Postcondition**    | Retention team has a prioritized call list for the week.                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Business Value**   | Proactive outreach to highest-risk customers before they cancel, reducing weekly churn.                                                                                                                                                                                                                                                                                                                                                                                                                             |

---

### UC-03: Model Performance Monitoring & Drift Detection

| Field                      | Detail                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Use Case ID**      | UC-03                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **Name**             | Model Performance Monitoring                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Actor**            | ML Engineer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Trigger**          | Grafana alert fires: "Prediction distribution shift detected in MonthlyCharges"                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Precondition**     | Prometheus metrics are being collected; Grafana alerting is configured                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **Main Flow**        | 1. ML Engineer receives alert notification.`<br>` 2. Engineer opens Grafana dashboard → confirms PSI score for `MonthlyCharges` > 0.2 (significant drift). `<br>` 3. Engineer triggers the retraining pipeline in GitHub Actions. `<br>` 4. New model is trained, evaluated, and validated against the baseline (must improve or match ROC-AUC). `<br>` 5. If validation passes, new Docker image is built and deployed automatically. `<br>` 6. MLflow registers the new model version as the production champion. |
| **Alternative Flow** | A2: If new model underperforms baseline, pipeline is halted and ML Engineer is notified.                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Postcondition**    | Production model is updated; prediction quality restored.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Business Value**   | Ensures the model remains accurate as the customer base and pricing evolve over time.                                                                                                                                                                                                                                                                                                                                                                                                                                            |

---

### UC-04: SHAP Explainability Report for Business Review

| Field                    | Detail                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Use Case ID**    | UC-04                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| **Name**           | Executive Churn Driver Analysis                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Actor**          | Business Analyst                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **Trigger**        | Monthly business review meeting                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **Precondition**   | Model has been running in production for at least 4 weeks                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| **Main Flow**      | 1. Analyst calls internal reporting endpoint to fetch global SHAP summary for the past month.`<br>` 2. System returns ranked list of top churn drivers: e.g., "1. Month-to-month contract (impact: +0.38), 2. Fiber Optic with no Online Security (+0.29), 3. Tenure < 12 months (+0.22)". `<br>` 3. Analyst presents findings in the review: "Fiber Optic customers without security add-ons are 2.4× more likely to churn." `<br>` 4. Business decision: promote "Fiber + Security Bundle" offer to at-risk segment. |
| **Postcondition**  | Business strategy is informed by model-derived insights, not just intuition.                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Business Value** | Converts ML outputs into actionable product/pricing decisions.                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

---

## 5. SCOPE & CONSTRAINTS

### 5.1. In Scope

| Area                          | Deliverable                                                                               |
| ----------------------------- | ----------------------------------------------------------------------------------------- |
| **Data**                | IBM Telco Customer Churn dataset only (static snapshot, no live API integration)          |
| **Modeling**            | Binary classification: Logistic Regression (baseline), XGBoost (primary), optional TabNet |
| **Explainability**      | SHAP global summary + per-prediction local explanation (reason codes)                     |
| **Serving**             | FastAPI REST API with `/predict`, `/predict/batch`, `/health` endpoints             |
| **Containerization**    | Dockerfile + Docker Compose for API, MLflow server, Prometheus, Grafana                   |
| **Monitoring**          | Prometheus metrics + Grafana dashboard + alerting rules                                   |
| **CI/CD**               | GitHub Actions: lint → test → build → validate model                                   |
| **Experiment Tracking** | MLflow: parameters, metrics, artifacts, model registry                                    |
| **Testing**             | Unit, integration, data quality, and model validation tests                               |

### 5.2. Out of Scope (This Iteration)

| What                                         | Why Excluded                                                                         |
| -------------------------------------------- | ------------------------------------------------------------------------------------ |
| Real-time streaming ingestion (Kafka, Flink) | Dataset is static; adds infrastructure complexity without demonstrating new ML value |
| Survival Analysis (time-to-churn)            | Telco dataset lacks event timestamps needed for proper survival modeling             |
| A/B testing framework                        | Requires production traffic; beyond scope of a course project                        |
| Multi-class churn classification             | Binary churn is sufficient for demonstrating all ML pipeline components              |
| Mobile/Web frontend UI                       | Scope is API-first; CRM integration is simulated via API calls                       |
| GDPR/CCPA compliance implementation          | Dataset is synthetic/anonymized; real compliance is out of academic scope            |

### 5.3. Technical Constraints

| Constraint           | Detail                                                           |
| -------------------- | ---------------------------------------------------------------- |
| **Data Size**  | Fixed at 7,043 records; no live data ingestion                   |
| **Compute**    | Must run on local machine (Docker Desktop); no cloud GPU assumed |
| **Language**   | Python 3.10+                                                     |
| **Framework**  | FastAPI for serving; no Flask or Django                          |
| **ML Library** | scikit-learn, XGBoost or LightGBM, SHAP                          |
| **Tracking**   | MLflow (local server via Docker)                                 |
| **Monitoring** | Prometheus + Grafana (via Docker Compose)                        |

### 5.4. Project Constraints

| Constraint           | Detail                                               |
| -------------------- | ---------------------------------------------------- |
| **Timeline**   | ~4 weeks (Presentation: Session 10)                  |
| **Team**       | 3–4 members; each must have meaningful Git commits  |
| **Submission** | Public GitHub repository; all required files present |
| **Deadline**   | April 18, 2026                                       |

---

*Document version: 1.0 | Created: 2026-04-15 | Course: DDM501 Final Project*
