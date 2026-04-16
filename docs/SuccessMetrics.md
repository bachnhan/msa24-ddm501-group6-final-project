# TELCO CUSTOMER CHURN PREDICTION SYSTEM
## Problem Definition & Requirements
### Section C — Success Metrics

**Course:** DDM501 - AI in Production: From Models to Systems
**Dataset:** IBM Telco Customer Churn
**Prepared by:** [Your Name]

---

## Overview

Success for this system is evaluated across **three hierarchical levels**. Each level has distinct owners, measurement cadences, and improvement levers:

```
┌─────────────────────────────────────────────────┐
│           BUSINESS METRICS (L1)                 │  Owner: VP Sales / Retention Manager
│   "Is the system creating business value?"      │  Cadence: Weekly / Monthly
├─────────────────────────────────────────────────┤
│            MODEL METRICS (L2)                   │  Owner: ML Engineer / Data Scientist
│   "Is the model learning correctly?"            │  Cadence: Per-experiment / Post-deploy
├─────────────────────────────────────────────────┤
│            SYSTEM METRICS (L3)                  │  Owner: ML Engineer / DevOps
│   "Is the system running reliably?"             │  Cadence: Real-time / Hourly
└─────────────────────────────────────────────────┘
```

---

## 1. BUSINESS METRICS (L1)

> These metrics answer: **"Is the ML system generating measurable business impact?"**
> Measured against a **no-model baseline** (random or rule-based targeting).

### 1.1. Primary Business Metrics

| Metric | Definition | Target | Measurement Method |
|---|---|---|---|
| **Churn Rate Reduction** | % decrease in monthly churn vs. pre-system baseline | ≥ 5% reduction within 3 months | CRM monthly churn reports |
| **Retention Campaign Precision** | % of contacted customers who were true churners | ≥ 65% | Outbound call outcome logs |
| **Retention Campaign Recall** | % of actual churners captured in the intervention list | ≥ 80% | Post-hoc churn audit vs. scored list |
| **Cost per Prevented Churn** | Retention spend ÷ number of churns prevented | Decrease vs. baseline | Finance + CRM reporting |

### 1.2. Revenue Impact Metrics

| Metric | Definition | Target | Notes |
|---|---|---|---|
| **Monthly Revenue at Risk (Identified)** | Sum of `MonthlyCharges` of all High-Risk customers (score ≥ 0.70) flagged by the model | Tracked weekly | Helps size the retention budget |
| **Revenue Retained** | Monthly revenue from customers who received intervention AND did not churn | ≥ $50,000/month (simulated on Telco data) | Estimated using average customer LTV × prevented churns |
| **Intervention ROI** | (Revenue retained − Retention spend) ÷ Retention spend | ≥ 3:1 | Discount cost vs. retained revenue |

### 1.3. Operational Efficiency Metrics

| Metric | Definition | Target |
|---|---|---|
| **False Intervention Rate** | % of retention offers sent to customers who would NOT have churned (false positives) | ≤ 35% |
| **High-Risk Tier Accuracy** | % of customers in the "High Risk" tier (score ≥ 0.70) who actually churned | ≥ 60% |
| **Top-Decile Lift** | Churn rate in top 10% scored customers ÷ overall churn rate | ≥ 3.0× |

> **Why Top-Decile Lift?** In a real Telco business, retention teams can realistically call only the top ~700 most at-risk customers per week (10% of 7,000). Lift quantifies how much better than random the model is at selecting that group.

### 1.4. Business Metrics Baseline (No-Model Scenario)

| Scenario | Churn Capture Rate | Wasted Interventions |
|---|---|---|
| **Random targeting** | ~26% (equal to base rate) | ~74% |
| **Rule-based** (e.g., "month-to-month + tenure < 12") | ~45% | ~55% |
| **ML Model (Target)** | ≥ 80% | ≤ 35% |

---

## 2. MODEL METRICS (L2)

> These metrics answer: **"Is the model producing accurate and trustworthy predictions?"**
> Evaluated on a stratified hold-out test set (20% of data, ~1,409 records).

### 2.1. Classification Performance

| Metric | Formula | Target | Why This Threshold |
|---|---|---|---|
| **ROC-AUC** | Area under the ROC curve | ≥ 0.85 | Strong discriminative ability; 0.5 = random, 1.0 = perfect |
| **Recall (Churn class)** | TP ÷ (TP + FN) | ≥ 0.80 | Priority metric: missing a churner is more costly than a false alarm |
| **Precision (Churn class)** | TP ÷ (TP + FP) | ≥ 0.65 | Avoid overwhelming retention team with false positives |
| **F1-Score (Churn class)** | 2 × (P × R) ÷ (P + R) | ≥ 0.72 | Harmonic balance of Precision and Recall |
| **PR-AUC** | Area under Precision-Recall curve | ≥ 0.68 | More informative than ROC-AUC under class imbalance |
| **Accuracy** | (TP + TN) ÷ Total | ≥ 0.78 | Secondary only; misleading under imbalance, not primary KPI |

> **Note on metric priority:** Given the class imbalance (~26.5% churn), **Recall** and **PR-AUC** are prioritized over Accuracy. A model predicting "No Churn" always achieves 73.5% accuracy but 0% Recall — exactly the failure pattern we guard against.

### 2.2. Confusion Matrix Targets (on Test Set, ~1,409 samples)

|  | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** (~1,034) | TN ≥ 671 (65%) | FP ≤ 363 (35%) |
| **Actual: Churn** (~375) | FN ≤ 75 (20%) | **TP ≥ 300 (80%)** ← Primary |

### 2.3. Model Calibration

| Metric | Definition | Target |
|---|---|---|
| **Expected Calibration Error (ECE)** | Mean absolute difference between predicted probability and actual frequency | ≤ 0.05 |
| **Reliability Diagram** | Predicted probabilities should closely track actual churn rates in each bucket | Visually within ±5% band |

> **Why calibration?** Business stakeholders trust scores like "this customer has an 82% chance of churning." If the model is poorly calibrated (e.g., predicts 80% but only 50% of those customers actually churn), the business cannot use the score to make financial decisions.

### 2.4. Model Comparison Table (Experiment Gate)

All candidate models are tracked in MLflow. A model is promoted to production only if it meets or exceeds the baseline on all primary metrics:

| Model | Expected ROC-AUC | Expected Recall | Expected F1 | Status |
|---|---|---|---|---|
| Logistic Regression | ~0.83 | ~0.75 | ~0.68 | Baseline |
| Random Forest | ~0.85 | ~0.78 | ~0.71 | Challenger |
| **XGBoost** ★ | **~0.88** | **~0.82** | **~0.74** | **Target Champion** |
| LightGBM | ~0.87 | ~0.81 | ~0.73 | Strong Alternative |
| TabNet (optional) | ~0.86 | ~0.80 | ~0.72 | Experimental |

### 2.5. Fairness Metrics

> Per the Responsible AI rubric requirement, model behavior must be analyzed across demographic groups present in the dataset.

| Metric | Groups Compared | Target |
|---|---|---|
| **Demographic Parity Difference** | Churn prediction rate: Female vs. Male | ≤ 0.05 |
| **Demographic Parity Difference** | Churn prediction rate: Senior Citizen vs. Non-Senior | ≤ 0.08 |
| **Equal Opportunity Difference** | Recall: Senior vs. Non-Senior | ≤ 0.07 |
| **SHAP Disparity** | Mean absolute SHAP value of `gender` and `SeniorCitizen` features | < 0.02 (low influence) |

### 2.6. Model Drift Detection Metrics (Post-Deployment)

| Metric | Trigger Threshold | Action |
|---|---|---|
| **Population Stability Index (PSI)** on `MonthlyCharges` | PSI > 0.20 = Significant drift | Trigger retraining pipeline |
| **PSI** on `Tenure` | PSI > 0.20 | Trigger retraining pipeline |
| **Prediction Score Distribution Shift** | Kolmogorov-Smirnov test p-value < 0.05 | Investigate + alert |
| **Rolling Recall** (if labels available) | Drop > 5% from deployment Recall | Investigate immediately |

---

## 3. SYSTEM METRICS (L3)

> These metrics answer: **"Is the serving infrastructure reliable, fast, and observable?"**
> Monitored continuously via Prometheus + Grafana.

### 3.1. API Performance

| Metric | Prometheus Metric Name | Target | Alert Threshold |
|---|---|---|---|
| **P50 Inference Latency** | `http_request_duration_seconds{quantile="0.5"}` | < 75ms | > 150ms |
| **P95 Inference Latency** | `http_request_duration_seconds{quantile="0.95"}` | < 150ms | > 300ms |
| **P99 Inference Latency** | `http_request_duration_seconds{quantile="0.99"}` | < 250ms | > 500ms |
| **Request Throughput** | `http_requests_total` rate | > 10 req/s sustained | < 1 req/s (suggests outage) |
| **Error Rate (4xx + 5xx)** | `http_errors_total / http_requests_total` | < 1% | > 5% over 5 min |

### 3.2. System Reliability

| Metric | Target | Alert Threshold |
|---|---|---|
| **API Uptime** | ≥ 99.5% (≤ 3.6 hrs/month downtime) | Service down > 5 minutes |
| **Container Restart Rate** | 0 unexpected restarts/day | Any unexpected restart |
| **Health Check Status** | `GET /health` returns 200 | Returns non-200 for > 1 min |
| **MLflow Tracking Availability** | ≥ 99% (local) | UI unreachable > 2 min |
| **Grafana Dashboard Availability** | ≥ 99% | Dashboard unreachable |

### 3.3. Resource Utilization

| Metric | Target (Normal Load) | Alert Threshold |
|---|---|---|
| **CPU Usage (API container)** | < 40% average | > 80% sustained > 5 min |
| **Memory Usage (API container)** | < 512MB | > 1GB |
| **SHAP Computation Time** | < 50ms per-prediction | > 200ms |
| **Batch Scoring (7,000 records)** | < 30 seconds | > 120 seconds |

### 3.4. CI/CD Pipeline Metrics

| Metric | Target |
|---|---|
| **CI Build Success Rate** | ≥ 95% of commits pass all checks |
| **Test Suite Execution Time** | < 3 minutes |
| **Code Coverage** | ≥ 80% on `src/` modules |
| **Docker Build Time** | < 5 minutes |
| **Model Validation Gate Pass Rate** | New model must match or beat baseline ROC-AUC |

---

## 4. METRICS DASHBOARD SUMMARY

The Grafana dashboard is organized into three panels mirroring the above hierarchy:

### Panel 1 — Business Pulse (Weekly View)
- Line chart: "High-Risk Customers Identified" over time
- Gauge: "Revenue at Risk this week" (sum of `MonthlyCharges` × churn probability for score ≥ 0.70)
- Bar chart: Retention campaign outcome (Converted / Refused / No contact)

### Panel 2 — Model Health (Per-Deploy View)
- Gauge: Current champion model ROC-AUC vs. deployment threshold (0.85)
- Line chart: Rolling Recall (7-day window, if feedback labels available)
- Heatmap: Prediction score distribution (should be stable week-over-week)
- PSI bar chart: Feature drift by variable

### Panel 3 — System Health (Real-Time)
- Time series: API P95 latency (red line at 150ms threshold)
- Counter: Total predictions served today / this week
- Error rate: Red/Green status indicator
- Docker service status: API ✅ | MLflow ✅ | Prometheus ✅ | Grafana ✅

---

## 5. METRICS TRACEABILITY MATRIX

> Maps each metric back to the stakeholder who owns it and the rubric criterion it satisfies.

| Metric | Level | Owner | Rubric Criterion |
|---|---|---|---|
| Churn Rate Reduction | Business | VP Sales | Business value demonstration |
| Top-Decile Lift | Business | Retention Manager | Business value demonstration |
| ROC-AUC | Model | ML Engineer | Model evaluation |
| Recall (Churn class) | Model | ML Engineer | Model evaluation |
| Fairness (Demographic Parity) | Model | ML Engineer | Responsible AI |
| PSI / Drift Score | Model | ML Engineer | Monitoring |
| API P95 Latency | System | DevOps | Deployment quality |
| Test Coverage | System | All Engineers | Testing & CI/CD |
| Container Uptime | System | DevOps | Deployment reliability |

---

*Document version: 1.0 | Created: 2026-04-15 | Course: DDM501 Final Project*
