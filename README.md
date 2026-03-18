# 🛡️ DriftGuard — Self-Healing ML Monitoring System

[![CI](https://github.com/YOUR_USERNAME/driftguard/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/driftguard/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.12-orange.svg)](https://mlflow.org)
[![Evidently](https://img.shields.io/badge/Evidently-0.4.30-purple.svg)](https://evidentlyai.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A **production-grade MLOps system** for credit card fraud detection that automatically
> detects data drift, retrains the model, and promotes the challenger only if it beats
> the champion — zero human intervention required.

🎯 **[Live Demo: Dashboard](https://YOUR_APP.streamlit.app)** |
📡 **[Live Demo: API Docs](https://driftguard-api.onrender.com/docs)**

---

## Architecture
[creditcard.csv] → [DVC] → [Feast Feature Store]
│
▼
[Prefect Pipeline] → [LightGBM+SMOTE] → [MLflow Registry (@champion alias)]
▲                                              │
│                                              ▼
[Self-Heal Loop] ← [Evidently AI Drift] ← [FastAPI /predict + aiosqlite log]
│
└── if drift → retrain → champion/challenger → promote if better
│
[Streamlit Dashboard: Predictions + Drift + Registry]
│
[GitHub Actions CI/CD] → [Render.com (API)] + [Streamlit Cloud (Dashboard)]

## Key Features

| Feature | Tool | What it does |
|---|---|---|
| Self-healing loop | Prefect 2.19 | Orchestrates detect → retrain → promote |
| Drift detection | Evidently 0.4.30 | Statistical drift on every production batch |
| Experiment tracking | MLflow 2.12 | Logs every run, aliases champion model |
| Model training | LightGBM 4.3 + SMOTE | Handles 0.17% fraud imbalance correctly |
| Feature store | Feast 0.39 | Prevents training-serving skew |
| Async serving | FastAPI + aiosqlite | Sub-10ms predictions, WAL-mode logging |
| Full stack | Docker Compose | One-command local deployment |
| CI/CD | GitHub Actions | Tests + auto-deploy on push to main |

## Quickstart
```bash
git clone https://github.com/YOUR_USERNAME/driftguard.git
cd driftguard
pip install -r requirements.txt

# 1. Download dataset → data/raw/creditcard.csv
#    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 2. Prepare data
make setup

# 3. Train model
make train

# 4. Run API (Terminal 1)
make api

# 5. Run dashboard (Terminal 2)
make dashboard
```

**Or use Docker (full stack):**
```bash
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
# MLflow:    http://localhost:5000
```

## Project Structure
driftguard/
├── app.py                              # Streamlit Cloud entrypoint
├── pages/                              # Streamlit multi-page app
│   ├── 1_predictions.py
│   ├── 2_drift_monitor.py
│   └── 3_model_registry.py
├── src/
│   ├── config.py                       # Central config (paths, params)
│   ├── data/preprocess.py              # Data splits + drift simulation
│   ├── features/feature_store.py       # Feast helpers
│   ├── training/
│   │   ├── train.py                    # LightGBM + SMOTE + MLflow
│   │   └── pipeline.py                 # Prefect orchestration flow
│   ├── serving/api.py                  # FastAPI + aiosqlite serving
│   └── monitoring/
│       ├── drift_detector.py           # Evidently AI reports
│       └── retrain_trigger.py          # Self-healing Prefect flow
├── feast/                              # Feature store definitions
│   ├── feature_store.yaml
│   └── features.py
├── tests/                              # pytest unit + integration tests
│   ├── conftest.py                     # Fixtures + CI skip markers
│   ├── test_config.py
│   ├── test_api.py
│   └── test_preprocessing.py
├── .github/workflows/
│   ├── ci.yml                          # Test + Docker build on every PR
│   └── deploy.yml                      # Auto-deploy to Render on main push
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml                  # Full local stack
├── render.yaml                         # Render.com IaC config
├── packages.txt                        # System deps for Streamlit Cloud
├── requirements.txt
└── Makefile

## Dataset

**Credit Card Fraud Detection** by ULB (Université Libre de Bruxelles)
284,807 transactions | 492 fraud (0.17%) | V1–V28 are PCA-transformed features
Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## License

MIT
