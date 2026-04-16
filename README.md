# CustomerChurn-Production: Predicting Customer Retention
[![CI/CD Pipeline](https://github.com/bachnhan/msa24-ddm501-group6-final-project/actions/workflows/ci.yml/badge.svg)](https://github.com/bachnhan/msa24-ddm501-group6-final-project/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C.svg?style=flat&logo=prometheus&logoColor=white)](https://prometheus.io)

## 📊 Project Overview

This project implements an end-to-end **Customer Churn Prediction System** for the **DDM501 - AI in Production** Capstone. Using industry-standard classification models (Random Forest), the system identifies customers at risk of leaving, enabling proactive retention strategies.

### 🎯 Problem Statement & Use Case
High customer churn rates directly impact profitability. Our system predicts the likelihood of churn based on behavioral patterns (usage frequency, support calls, payment delays), allowing marketing teams to offer targeted incentives to high-risk customers perfectly aligned with the [Kaggle Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset).

## 🏗️ System Design & Architecture

Our system is built for **Scalability**, **Reliability**, and **Observability**. We follow a microservices-inspired architecture containerized with Docker.

### High-Level Architecture
```mermaid
graph TD
    User([CRM / Business App]) -->|POST /predict| LB[FastAPI Gateway]
    subgraph "ML Inference Container"
        LB --> PP[Preprocessor]
        PP --> RF[Random Forest Model]
        RF --> GU[Guardrails]
    end
    LB -.->|Prometheus| LOGS[(TSDB)]
    LOGS --> GRAF[Grafana Dashboards]
```

### Inference Data Flow
1. **Request**: Client sends customer JSON features.
2. **Validation**: Pydantic ensures data integrity.
3. **Pipeline**: Scikit-Learn Pipeline handles scaling and one-hot encoding.
4. **Prediction**: Ensemble model generates churn probability.
5. **Guardrail**: System verifies output constraints and logs to Prometheus.

For detailed technology justifications and trade-off analysis, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 📂 Project Structure
```text
.
├── app/
│   ├── main.py             # FastAPI churn endpoints
│   ├── model.py            # RF Classifier wrapper
│   ├── schemas.py          # Customer feature schemas
├── scripts/
│   ├── train_model.py      # Classification training (MLflow)
│   ├── explain_model.py    # Feature importance analysis
│   └── fairness_analysis.py# Bias detection (Gender/Tenure)
├── models/                 # Model pipeline (.pkl)
└── ...
```

---

## 🚀 Getting Started

### 1. Initial Setup
```bash
git clone git@github.com:bachnhan/msa24-ddm501-group6-final-project.git
cd msa24-ddm501-group6-final-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Model Training
```bash
# Trains Random Forest and logs to MLflow
python scripts/train_model.py
```

### 3. Usage & Explanations
```bash
# Start API
uvicorn app.main:app --reload

# Explain predictions
python scripts/explain_model.py
```

---

## ⚖️ Responsible AI
- **Explainability**: Uses Global Feature Importance to identify top churn drivers.
- **Fairness**: Monitors error rate parity between different user demographics.
- **Guardrails**: Validates input ranges (e.g., age, tenure) to prevent out-of-distribution errors.

---
© 2026 DDM501 Group 6 - AI in Production
