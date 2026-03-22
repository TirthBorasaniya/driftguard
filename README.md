# DriftGuard: Self-Healing ML Monitoring System (IEEE-CIS Edition)

[![CI](https://github.com/YOUR_USERNAME/driftguard/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/driftguard/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.12-orange.svg)](https://mlflow.org)
[![Evidently](https://img.shields.io/badge/Evidently-0.4.30-purple.svg)](https://evidentlyai.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)

A production-grade MLOps system for e-commerce fraud detection on the
IEEE-CIS dataset (590k transactions, Vesta Corporation). Detects genuine
temporal data drift, retrains automatically, and promotes a challenger model
only when it outperforms the current champion.

**Live demo:** [Dashboard](https://YOUR_APP.streamlit.app) |
[API](https://driftguard-ieee-api.onrender.com/docs)

---

## Architecture
train_transaction.csv + train_identity.csv
|
v
Left join on TransactionID -> Feature engineering -> Temporal split
|                         (log-transform,         (train/test/ref/
|                          D normalisation,         batches by time)
|                          label encoding)
|
v
DVC (data versioning) -> Feast (feature store, Parquet offline + SQLite online)
|
v
Prefect Pipeline -> LightGBM + SMOTE -> MLflow Registry (@champion alias)
^                                          |
|                                          v
Self-Healing Loop <- Evidently AI Drift <- FastAPI /predict + aiosqlite log
(real temporal drift)
|
Streamlit Dashboard (Predictions, Drift, Registry)
|
GitHub Actions CI/CD -> Render.com + Streamlit Cloud

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| Left join transaction + identity | 59% of transactions have no identity record; inner join would lose them |
| Temporal split, not random split | Random split causes data leakage in time-ordered fraud data |
| D column normalisation (Dn = D - day) | Removes absolute time trend; makes feature stable across time windows |
| log1p(TransactionAmt) | Reduces right skew in the transaction amount distribution |
| scale_pos_weight=28 | Reflects 3.5% fraud rate (96.5/3.5 = 27.6) without inflating dataset |
| Label encoding, not one-hot | LightGBM handles ordinal-encoded categoricals natively and efficiently |
| Encoders committed to repo | Render free tier has ephemeral filesystem; artefacts must be in the image |
| Real temporal drift for monitoring | Batches from later time windows have genuine distributional shift |

## Dataset

IEEE-CIS Fraud Detection - Kaggle / Vesta Corporation / IEEE Computational Intelligence Society
590,540 e-commerce transactions | 20,663 fraud cases (3.5%) | ~96 engineered features
Download: https://www.kaggle.com/competitions/ieee-fraud-detection/data

## Quickstart
```bash
git clone https://github.com/YOUR_USERNAME/driftguard.git
cd driftguard
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Place train_transaction.csv and train_identity.csv in data/raw/
make setup     # ~5 minutes - joins, engineers features, splits temporally
make train     # ~10 minutes - trains LightGBM on 413k rows
make api       # Terminal 1
make dashboard # Terminal 2
```

Full stack:
```bash
docker compose up --build
```

## License
MIT
