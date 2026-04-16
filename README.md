# MovieLens-Production: End-to-End Movie Recommendation System
[![CI/CD Pipeline](https://github.com/bachnhan/msa24-ddm501-group6-lab2/actions/workflows/ci.yml/badge.svg)](https://github.com/bachnhan/msa24-ddm501-group6-lab2/actions)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C.svg?style=flat&logo=prometheus&logoColor=white)](https://prometheus.io)

## 🎥 Project Overview

This project implements a production-ready **Movie Recommendation System** as the final capstone for **DDM501 - AI in Production**. We leverage Collaborative Filtering (SVD) to provide real-time rating predictions for users, integrated into a robust MLOps ecosystem including monitoring, experiment tracking, and automated CI/CD.

### 🎯 Problem Statement
In the modern streaming era, information overload prevents users from finding content they enjoy. Our system aims to increase user engagement by predicting movie ratings with high accuracy and low latency, enabling personalized content discovery.

---

## 🏗️ System Architecture

Our system is designed for scalability and observability, utilizing a microservices approach:

```mermaid
graph TD
    User([User/Client]) -->|REST API| FastAPI[FastAPI Model Service]
    FastAPI -->|Load Model| Pickle[Trained Model .pkl]
    FastAPI -->|Expose Metrics| Prometheus[Prometheus]
    Prometheus -->|Collect Metrics| Grafana[Grafana Dashboards]
    FastAPI -->|Log Experiments| MLflow[MLflow Tracking]
    
    subgraph "CI/CD Pipeline (GitHub Actions)"
        Lint[Linting / Formatting] --> Tests[Unit & Integration Tests]
        Tests --> Build[Docker Build & Push]
    end
```

For detailed design decisions and trade-offs, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 🚀 Key Features

### 1. ML Pipeline & Serving
- **Algorithm**: SVD (Singular Value Decomposition) from the Surprise library.
- **REST API**: FastAPI with asynchronous endpoints for single and batch predictions.
- **Experiment Tracking**: Integrated with **MLflow** to track hyperparameters and model artifacts.

### 2. MLOps & Monitoring
- **Containerization**: Full stack deployment using Docker Compose.
- **Observability**: 
    - **Prometheus**: Scrapes custom system and ML-specific metrics.
    - **Grafana**: Visualizes prediction distributions, error rates, and system latency.
    - **Alerting**: Real-time alerts via Prometheus Alertmanager for high latency or model failures.

### 3. Testing & CI/CD
- **Unit Testing**: 90%+ coverage on core model logic and API endpoints.
- **Automation**: GitHub Actions pipeline for automated linting, testing, and container builds.
- **Data Quality**: Integrated data validation tests to ensure input integrity.

### 4. Responsible AI
- **Explainability**: Implementation of global feature importance insights.
- **Fairness**: Bias analysis across user demographics to ensure equitable recommendations.
- **Security**: Data anonymization and secure model serving practices.

---

## 📥 Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.9+ (for local development)

### Quick Launch (Docker)
The easiest way to run the entire stack:

```bash
# Clone the repository
git clone https://github.com/bachnhan/msa24-ddm501-group6-lab2.git
cd msa24-ddm501-group6-lab2

# Start all services
docker-compose up -d
```

### Service Access Links
| Service | URL | Note |
|:--- |:--- |:--- |
| **API Documentation** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive Swagger UI |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | Query raw metrics |
| **Grafana** | [http://localhost:3000](http://localhost:3000) | Dashboards (admin/admin) |
| **MLflow** | [http://localhost:5000](http://localhost:5000) | Experiment tracking |

---

## 🛠️ Individual Contributions

| Member | Primary Responsibilities |
|:--- |:--- |
| **Lê Huỳnh Trang** | Problem Definition, ML Pipeline, Responsible AI (Bias/Fairness) |
| **Đỗ Trọng Minh Quân** | Requirements, Model Training, MLflow Integration, Documentation |
| **Nguyễn Huỳnh Bách Nhân** | System Architecture, Docker/API, Monitoring Stack, CI/CD Pipeline |

For detailed roles, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📚 Documentation
- [Architecture & Design Decisions](ARCHITECTURE.md)
- [API Specification (OpenAPI)](app/static/openapi.json)
- [Responsible AI Report](reports/responsible_ai.md)
- [Deployment Guide](docs/deployment.md)

---
© 2026 DDM501 Group 6 - AI in Production
