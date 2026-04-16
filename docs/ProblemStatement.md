# TELCO CUSTOMER CHURN PREDICTION SYSTEM
## Problem Definition & Requirements
### Section A — Problem Statement & Business Context

**Course:** DDM501 - AI in Production: From Models to Systems
**Topic:** Customer Churn Prediction (Focus: Telecommunications Industry)
**Dataset:** IBM Telco Customer Churn
**Prepared by:** [Your Name]

---

## 1. BUSINESS CONTEXT

### 1.1. Industry Background

In the highly competitive **Telecommunications (Telco) industry**, customer churn — the rate at which subscribers cancel their services — is a major barrier to growth. The cost of acquiring a new subscriber in the telecom sector is often significantly higher (up to **5-10 times**) than the cost of retaining an existing one.

Telco companies face unique challenges:
- **Price Wars:** Competitors constantly offer "switch and save" promotions.
- **Service Quality:** Issues with internet speed or technical support lead to immediate dissatisfaction.
- **Contract Rigidness:** Customers often churn at the end of month-to-month contracts to avoid long-term commitments.

Research shows that even a small reduction in churn (e.g., **2-5%**) can lead to a substantial increase in profit, as retained customers tend to adopt more services (Internet, Streaming, Security) over time, increasing their total Customer Lifetime Value (CLV).

### 1.2. Strategic Motivation

This project aims to move the company from **Reactive** to **Proactive** retention:

| Business Need | Reactive (Status Quo) | Proactive (Proposed ML System) |
|---|---|---|
| Churn Detection | When the customer cancels the contract. | Signs detected 30-60 days before risk. |
| Outreach Strategy | Standard "Come back" emails. | Personalized offers (e.g., DSL to Fiber upgrade). |
| Resource Allocation | Distributed equally across all customers. | Focused on "High-Value, High-Risk" segments. |

---

## 2. PROBLEM STATEMENT

### 2.1. Core Problem

> **How can we accurately identify individual subscribers at risk of churning within the next 1-3 billing cycles based on their service usage, contract attributes, and demographics?**

Key challenges specific to the Telco dataset:

#### (a) Feature Interaction (Service Bundling)
The Telco dataset contains binary and categorical features (Internet Service, Online Security, Tech Support). Understanding how the *absence* of certain support features (like Tech Support) interacts with high Monthly Charges to drive churn is critical for explainability.

#### (b) Managing Imbalance (~26.5% Churn)
With a churn rate of roughly 26%, the dataset is imbalanced. A model that always predicts "Not Churn" would still be 74% accurate but would be useless for the business. We must optimize for **Recall** to ensure no potential "leaks" are missed.

#### (c) Data Type Heterogeneity
The dataset includes `TotalCharges` (numeric), `Tenure` (numeric), and many categorical variables. Handling these via robust preprocessing pipelines (handling ' ' in TotalCharges, One-Hot Encoding) is a prerequisite for production.

### 2.2. Problem Scope

**In Scope:**
- Binary classification: `Churn` (Yes/No).
- Pipeline: Ingestion → Preprocessing → Modeling → SHAP Explainability → API Serving.
- Monitoring: Tracking distribution shifts in `MonthlyCharges` and `Tenure`.

**Out of Scope:**
- Predicting *how long* until they churn (Survival Analysis time-to-event) — although suggested in the topic, this iteration focuses on the classification of the immediate risk.
- Handling real-time streaming usage data (only snapshot data is used).

### 2.3. Target Dataset: Telco Customer Churn (IBM)

| Attribute | Details |
|---|---|
| **Size** | 7,043 customers |
| **Target Variable** | `Churn` (Yes: 1,869 | No: 5,174) |
| **Key Demographics** | Gender, Senior Citizen, Partner, Dependents |
| **Service Features** | Phone, Multiple Lines, Internet (DSL/Fiber), Security, Backup, Streaming |
| **Contractual** | Contract (Month-to-month, One year, Two year), Paperless Billing, Payment Method |
| **Financials** | Tenure (months), Monthly Charges, Total Charges |

---

## 3. SUCCESS METRICS

### 3.1. Business Metrics
- **Recall (Churn Class) ≥ 0.80:** We want to capture 80% of people who would have actually left.
- **Top-Decile Lift:** The model should be at least 3x better than random at picking churners in its highest-risk group.
- **Intervention Cost Efficiency:** Reduction in "wasted" discounts sent to customers who were never going to churn.

### 3.2. Model & System Metrics
- **ROC-AUC ≥ 0.85**
- **F1-Score ≥ 0.70**
- **Inference Latency < 150ms:** Fast enough to be embedded in a CRM dashboard or Call Center agent UI.

---

## 4. PROPOSED SOLUTION OVERVIEW

The system will be built as a modular ML project:
1. **Data Layer:** Cleanse Telco raw data, handle missing `TotalCharges`, and version with DVC or simple artifact tracking.
2. **Experimentation:** Compare **XGBoost** and **LightGBM** (excellent for tabular data) using MLflow.
3. **Serving:** A **FastAPI** wrapper that takes a customer ID or feature JSON and returns a risk score + "Reason for Risk" (via SHAP).
4. **Monitoring:** A Grafana dashboard showing the "Most Important Churn Drivers" this week vs. last week.

---

## 5. USE CASE: CALL CENTER RETENTION

**Scenario:** A customer calls to check their balance.
1. The CRM system automatically pings the **Churn Prediction API**.
2. If the score is **> 0.7**, a "High Risk" alert pops up on the agent's screen.
3. **Explainability Insight:** API returns "Reason: Month-to-month contract & No Tech Support".
4. **Action:** The agent is authorized to offer a 20% discount if the customer switches to a "One-Year Contract" with "Free Tech Support".

---
*Document version: 1.1 | Updated for Telco Industry Focus | 2026-04-15*
