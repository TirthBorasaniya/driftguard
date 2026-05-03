# DriftGuard — Real-Time Fraud Detection Pipeline

[![CI](https://github.com/TirthBorasaniya/driftguard/actions/workflows/ci.yml/badge.svg)](https://github.com/TirthBorasaniya/driftguard/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LightGBM](https://img.shields.io/badge/model-LightGBM-green)
![Kafka](https://img.shields.io/badge/streaming-Kafka-orange)

End-to-end production ML pipeline for real-time credit card fraud detection. Kafka event ingestion, Redis-backed Feast feature store, LightGBM with F2-optimized threshold, SHAP explanations, Evidently AI drift detection with automatic self-healing retraining, and Prometheus + Grafana infrastructure monitoring. Full Docker Compose stack.

---

## Model Performance

Trained on the [Sparkov synthetic credit card dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (1.85M transactions, 0.58% fraud rate).

| Metric | Value |
|---|---|
| AUC-ROC | 0.9751 |
| AUC-PR | 0.6096 |
| F2-Score | 0.6760 |
| Precision | 0.6148 |
| Recall | 0.6933 |
| Decision threshold | 0.9646 (F2-optimized) |

The threshold is optimized at the F2 operating point on the Precision-Recall curve. F2 weights recall twice as heavily as precision, which is the correct tradeoff for fraud detection where missing fraud is more costly than false alarms.

---

## Architecture

```
fraudTrain.csv
      |
      v
[Preprocessing]  ------>  data/processed/  (train, test, reference, stream splits)
      |                         |
      v                         v
[Great Expectations]     [Feast Offline Store]
  Data validation          Feature aggregation
  (abort on failure)       (card 7-day stats,
                            category fraud rate)
      |                         |
      v                         v
[LightGBM Training]      [Redis Online Store]
  scale_pos_weight=172     Sub-ms feature lookup
  F2 threshold opt.        at serving time
  MLflow tracking
  Champion/Challenger
      |
      v
[FastAPI Serving]  <---  Kafka Consumer  <---  Kafka Producer
  /predict                Pydantic validation    Chronological replay
  /predict/explain        DLQ for malformed      --drift flag for
  SHAP top-5              Manual offset commit     demo mode
  Prometheus /metrics     Feast online lookup
      |
      v
[Evidently Drift Detector]
  500-event window guard
  KS test + chi-square
  StreamingDriftDetector
      |
   Drift?
      |
      v
[Prefect Retraining Flow]
  Step 1: GE validation (abort on failure)
  Step 2-4: Load data, materialize features
  Step 5: Train challenger
  Step 6-7: Evaluate, compute F2 threshold
  Step 8: Compare vs champion
  Step 9: Promote + hot-reload API
      |
      v
[Prometheus + Grafana]
  infra.json: latency, error rate, throughput
  ml_health.json: drift score, AUC-PR, DLQ rate
```

---

## Stack

| Layer | Tool | Why |
|---|---|---|
| Streaming | Kafka (confluent-kafka) | At-least-once delivery, manual offset commit, DLQ |
| Feature store | Feast 0.40 + Redis 7 | Training-serving consistency, sub-ms online lookup |
| Data validation | Great Expectations | Abort retraining on bad data before it reaches the model |
| Model | LightGBM 4.x | Best tabular performance; native categorical handling |
| Class imbalance | `scale_pos_weight=172` | No SMOTE — avoids interpolation artifacts on high-cardinality categoricals |
| Threshold | F2 on Precision-Recall curve | Correct operating point for fraud detection |
| Experiment tracking | MLflow 2.12 | Champion/challenger alias promotion |
| Orchestration | Prefect 2.x | 9-step retraining flow with GE validation gate |
| Drift detection | Evidently AI 0.4.30 | KS test on features, 500-event minimum window guard |
| Serving | FastAPI + Uvicorn | Async, lifespan model hot-reload, background task logging |
| Explainability | SHAP TreeExplainer | Regulatory requirement in production fraud systems |
| Monitoring | Prometheus + Grafana | Industry-standard ops stack; two provisioned dashboards |
| Encoder | Custom SafeLabelEncoder | -1 fallback for unseen categories; sklearn raises ValueError |
| CI | GitHub Actions | ruff + mypy + pytest on every push |

---

## Drift Detection and Self-Healing

The pipeline handles drift in two ways:

**Natural temporal drift:** The stream split (last 5% of data by time) is replayed chronologically. The 24-month Sparkov window contains real seasonal patterns that fire Evidently's KS test without injection.

**Drift injection mode:** For controlled demos, the producer supports a `--drift` flag that applies `data/drift_scenarios/heavy_drift.json`:
- Transaction amounts scaled 3x
- Category distribution concentrated to `grocery_pos` and `gas_transport`
- 70% of transactions routed to Texas

This fills the 500-event window in roughly 2 minutes and triggers the full self-healing cycle, visible in Grafana in real time.

**Self-healing cycle:**
1. `StreamingDriftDetector` accumulates 500 events, runs Evidently report
2. On drift breach: `alert_handler.py` triggers `retraining_flow`
3. 9-step Prefect flow: GE validation gate → train challenger → compare → promote
4. API detects `production_model.pkl` mtime change and hot-reloads without restart

---

## Project Structure

```
driftguard/
├── src/
│   ├── config.py                     # pydantic-settings: all env vars, paths, feature lists
│   ├── data/
│   │   ├── encoders.py               # SafeLabelEncoder with -1 fallback for unseen categories
│   │   └── preprocess.py             # Feature engineering, temporal split at P75
│   ├── features/
│   │   ├── engineering.py            # Single source of truth for transforms (training + serving)
│   │   ├── feature_repo/             # Feast: cc_num entity, card_stats_7d, category_fraud_rate
│   │   └── materializer.py           # Push offline features to Redis online store
│   ├── validation/
│   │   └── expectations.py           # GE suite: amt, city_pop, cardinality, fraud rate, uniqueness
│   ├── training/
│   │   ├── train.py                  # LightGBM, MLflow, champion/challenger
│   │   ├── evaluate.py               # AUC-PR, AUC-ROC, F2 at threshold
│   │   └── threshold.py              # F2 operating point on Precision-Recall curve
│   ├── serving/
│   │   ├── main.py                   # FastAPI lifespan, Prometheus, hot-reload loop
│   │   ├── routes.py                 # /predict, /predict/explain, /health
│   │   ├── schemas.py                # Pydantic request/response models
│   │   ├── model_loader.py           # Load champion model + threshold + encoders
│   │   └── explainer.py              # SHAP TreeExplainer, top-5 contributions
│   ├── producer/
│   │   ├── kafka_producer.py         # Chronological stream replay
│   │   └── drift_injector.py         # Configurable distributional shift
│   ├── consumer/
│   │   ├── kafka_consumer.py         # Validate, infer, log, drift-detect, commit offset
│   │   ├── schemas.py                # Pydantic TransactionEvent
│   │   └── dlq_handler.py            # Route malformed messages to transactions.dlq
│   ├── monitoring/
│   │   ├── drift_detector.py         # StreamingDriftDetector with 500-event guard
│   │   └── alert_handler.py          # Trigger retraining on drift breach
│   └── orchestration/flows/
│       └── retraining_flow.py        # 9-step Prefect flow
├── monitoring/grafana/
│   ├── dashboards/infra.json         # Request rate, latency, error rate
│   └── dashboards/ml_health.json     # Drift score, AUC-PR, DLQ rate, prediction distribution
├── benchmark/locustfile.py           # Locust load test for /predict
├── tests/                            # pytest: unit tests, no data required for CI
├── docker-compose.yml                # Kafka, Zookeeper, Redis, API, Prometheus, Grafana
└── Dockerfile.api
```

---

## Quick Start

### Prerequisites

- Python 3.11
- Docker Desktop (for full stack)
- Kaggle account (for dataset)

### 1. Setup

```bash
conda create -n driftguard python=3.11 -y
conda activate driftguard
pip install -r requirements.txt
```

### 2. Download dataset

Download `fraudTrain.csv` and `fraudTest.csv` from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) into `data/raw/`.

### 3. Run the pipeline

```bash
# Preprocess: engineer features, encode, temporal split
python -m src.data.preprocess

# Feast: register feature views and materialize to Redis
cd src/features/feature_repo && feast apply && cd ../../..
python -m src.features.materializer

# Train: LightGBM, F2 threshold, MLflow, champion promotion
python -m src.training.train

# Serve API
uvicorn src.serving.main:app --reload --port 8000

# MLflow UI (separate terminal)
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

### 4. Run full Docker stack

```bash
cp .env.example .env   # set GF_ADMIN_PASSWORD
docker compose up --build
```

| Service | URL |
|---|---|
| Fraud Detection API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### 5. Stream transactions and trigger drift

```bash
# Normal mode: replay stream split chronologically
python -m src.producer.kafka_producer

# Consumer: validate, infer, log, detect drift (separate terminal)
python -m src.consumer.kafka_consumer

# Drift injection mode: fires detection in ~2 minutes
python -m src.producer.kafka_producer --drift
```

### 6. Manually run retraining flow

```bash
python -m src.orchestration.flows.retraining_flow
```

---

## API Reference

```bash
# Health check
curl http://localhost:8000/health

# Predict fraud probability
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cc_num": "4532015112830366",
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "misc_net",
    "amt": 149.62,
    "gender": "F",
    "city": "Henderson",
    "state": "TX",
    "zip": "76054",
    "lat": 36.0788,
    "long": -81.1781,
    "city_pop": 35550,
    "job": "Scientist",
    "dob": "1987-01-01",
    "merch_lat": 36.011,
    "merch_long": -82.048,
    "trans_date_trans_time": "2020-06-21 12:14:25"
  }'

# Predict with SHAP explanation (top-5 feature contributions)
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{ ...same payload... }'
```

**Example `/predict` response:**
```json
{
  "transaction_id": "a3f9c2d1e4b7",
  "fraud_probability": 0.031,
  "is_fraud": false,
  "threshold": 0.9646,
  "model_version": "20260429-143021",
  "timestamp": "2026-04-29T14:30:21.432Z"
}
```

**Example `/predict/explain` response:**
```json
{
  "fraud_probability": 0.031,
  "is_fraud": false,
  "threshold": 0.9646,
  "top_features": [
    {"feature": "distance_km", "shap_value": -0.412},
    {"feature": "amt_log",     "shap_value": -0.183},
    {"feature": "hour_of_day", "shap_value":  0.092},
    {"feature": "category",    "shap_value": -0.071},
    {"feature": "city_pop",    "shap_value": -0.044}
  ]
}
```

---

## Tests

```bash
# CI (no dataset required)
pytest tests/ -m "not requires_data" -v

# Full suite (requires fraudTrain.csv)
pytest tests/ -v

# Lint and type check
ruff check src/ tests/
mypy src/
```

---

## Latency Benchmark

```bash
locust -f benchmark/locustfile.py --headless -u 50 -r 10 --run-time 60s --host http://localhost:8000
```

---

## Key Design Decisions

**No SMOTE.** `scale_pos_weight=172` handles the 99.42%/0.58% class split. SMOTE on high-cardinality categorical features interpolates in a space where interpolation is semantically invalid and introduces leakage risk if applied before splitting.

**SafeLabelEncoder over sklearn LabelEncoder.** sklearn raises `ValueError` on unseen categories at inference time. `SafeLabelEncoder` maps them to `-1`, which LightGBM treats as a distinct bin. This prevents API crashes when new merchants or states appear in production.

**F2 threshold over default 0.5.** The threshold is computed at the F2 operating point on the Precision-Recall curve after each training run and stored as `models/threshold.json`. It is recomputed on every retraining cycle.

**500-event drift window guard.** Evidently's KS test is statistically unreliable on N < 500. The `StreamingDriftDetector` accumulates events silently below this threshold and logs a warning rather than running a spurious report.

**champion_metrics.json over MLflow API.** The retraining flow reads `models/champion_metrics.json` for the champion/challenger comparison rather than calling the MLflow tracking server. This eliminates a network dependency from the critical path.

**engineering.py as single source of truth.** All feature transforms live in `src/features/engineering.py` and are called identically at training time and serving time. Any divergence causes training-serving skew, which is the most common source of production ML degradation.

---

## Resume Bullets

- Built real-time fraud detection pipeline processing Kafka event streams through Feast feature store (Redis online store) into LightGBM classifier with F2-optimized threshold (AUC-PR 0.61, Recall 0.69) and FastAPI serving with SHAP explanations at sub-100ms latency
- Implemented self-healing retraining loop using Prefect: Great Expectations validation gate, Evidently AI windowed drift detection (500-event KS test guard), and automated MLflow champion/challenger promotion triggered on distributional shift
- Deployed full observability stack with Prometheus and Grafana (two provisioned dashboards), custom metrics for drift score, DLQ rate, and champion AUC-PR; at-least-once Kafka delivery with dead letter queue for malformed events

---

## License

MIT
