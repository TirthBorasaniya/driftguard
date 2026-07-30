# DriftGuard — Real-Time Network Anomaly Detection Pipeline

[![CI](https://github.com/TirthBorasaniya/driftguard/actions/workflows/ci.yml/badge.svg)](https://github.com/TirthBorasaniya/driftguard/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![LightGBM](https://img.shields.io/badge/model-LightGBM-green)
![Kafka](https://img.shields.io/badge/streaming-Kafka-orange)

End-to-end production ML pipeline for network telemetry anomaly detection on CICIDS2017 network flow data. Kafka event ingestion, Redis-backed Feast feature store, LightGBM evaluated by PR-AUC with a recall-calibrated serving threshold, SHAP explanations, Evidently AI drift detection with automatic self-healing retraining, and Prometheus + Grafana infrastructure monitoring. Full Docker Compose stack.

---

## Architecture

```mermaid
flowchart TD
    A[CICIDS2017 Replay Producer] -->|Avro via Schema Registry| B[Kafka: network_flows]
    B --> C{Schema Valid?}
    C -->|No| D[Kafka: network_flows.dlq]
    C -->|Yes| E[Redis: Idempotent Dedup Check]
    E -->|Duplicate| F[Skip + Commit Offset]
    E -->|New| G[Feature Engineering]
    G --> H[Feast Online Store]
    G --> I[Feast Offline Store]
    H --> J[FastAPI Serving: LightGBM Champion]
    J --> K[Predictions + SHAP Explanations]

    G --> L[Evidently: drift_share Monitor]
    G --> M[PSI Drift Detector vs Monday Baseline]
    L --> N{Drift Detected?}
    M --> N
    N -->|Yes| O[Prefect Retraining Flow]
    O --> P[Great Expectations: Bounded Validation]
    P -->|Fail| Q[Abort Retraining]
    P -->|Pass| R[Train Challenger Model]
    R --> S{AUC-PR Margin Exceeded?}
    S -->|Yes| T[MLflow: Promote via Alias]
    S -->|No| U[Hold Champion]
    T --> J

    I --> V[Point-in-Time Offline Materialization]

    style D fill:#f8d7da
    style Q fill:#f8d7da
    style U fill:#fff3cd
    style T fill:#d4edda
```

`drift_share` (Evidently's `DatasetDriftMetric`) and PSI (`compute_psi_scores` against the
benign Monday reference) are two independent signals, either of which can trigger the
Prefect retraining flow (`check_drift_and_trigger_retraining` for the PSI path). Malformed
events are routed to the DLQ by `dlq_handler.route_to_dlq`; the dedup step is
`dedup.is_duplicate_event`, backed by Redis `SETNX`. Not pictured: shadow-mode scoring
(`shadow_mode.score_shadow_mode`), which runs a registered challenger alongside the
champion on live traffic and logs both predictions without serving the challenger's.

---

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset 2017): **2,830,540** labeled network flow records across five daily capture files, each row carrying ~80 flow features. The binary target is `0 = BENIGN` versus `1 = any attack` (DoS, DDoS, PortScan, Brute Force, Web Attacks, Infiltration, Bot, Heartbleed).

Capture files, in replay (chronological) order:

1. `Monday-WorkingHours.pcap_ISCX.csv` (benign only — drift baseline)
2. `Tuesday-WorkingHours.pcap_ISCX.csv`
3. `Wednesday-workingHours.pcap_ISCX.csv`
4. `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
5. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

Place all files under `data/raw/cicids2017/`.

**Limitations.** CICIDS2017 is the field-standard benchmark for network intrusion
detection, but it is not flawless: Liu et al., "Error prevalence in NIDS datasets: a
case study on CIC-IDS-2017 and CSE-CIC-IDS-2018," documents labeling inconsistencies
and flow-construction errors in the CICFlowMeter-generated features (including
mislabeled benign/attack flows and duplicate or malformed flow records) present in
the released CSVs. This project uses the dataset as-is despite these known issues
because it remains the de facto standard for demonstrating NIDS pipelines, not
because the labels or features are considered ground truth. Directly encountered in
this project: a small number of real rows (3 out of 647,909 in the training split, a
documented CICFlowMeter clock-sync capture artifact) had negative `flow_duration` and
related timing/throughput values; `filter_invalid_flow_rows` in `src/data/preprocess.py`
drops them at the source and logs the exact count and reason.

---

## Features

The model is trained on ten flow features (nine pass-through plus one engineered), defined once in `src/features/engineering.py` and computed identically at training and serving time:

| Feature | Source | Captures |
|---|---|---|
| `flow_duration` | pass-through | total flow lifetime |
| `flow_bytes_per_sec` | pass-through | byte throughput |
| `flow_packets_per_sec` | pass-through | packet throughput |
| `total_fwd_packets` | pass-through | forward packet volume |
| `total_bwd_packets` | pass-through | backward packet volume |
| `packet_length_mean` | pass-through | mean payload length |
| `packet_length_std` | pass-through | payload length spread |
| `flow_iat_mean` | pass-through | mean inter-arrival time |
| `fwd_bwd_packet_ratio` | engineered | directional asymmetry: `total_fwd_packets / (total_bwd_packets + 1e-9)` |
| `syn_flag_count` | pass-through | SYN flags, elevated in port scans and SYN floods |

The entity key is `src_ip` (source IP address).

---

## Operating Point

The primary evaluation metric is **PR-AUC** (average precision), the standard metric for severe class imbalance: it summarizes performance across all thresholds without committing to a single operating point during training. Threshold-specific metrics such as Fbeta are sensitive to the chosen cutoff and misleading under extreme imbalance.

For serving, the decision threshold is **calibrated to achieve a minimum recall of 0.95** on the validation split (a missed attack is more costly than a false alarm), maximizing precision subject to that recall floor. Threshold calibration runs after training, not during.

A challenger is promoted to champion only when its PR-AUC exceeds the champion's by more than `0.005`. Metrics and the calibrated threshold are produced by `python -m src.training.train` on the processed CICIDS2017 splits.

**Real, verified results** (final run against the five real CICIDS2017 capture files, after tuning `scale_pos_weight` dynamically from the true training class ratio and filtering the small number of CICFlowMeter clock-sync artifact rows): **PR-AUC 0.1625**, ROC-AUC 0.9383, calibrated threshold 0.0298, **realized recall 95.6%**, **realized precision 17.2%**. Verified end to end against a live Docker Compose stack and a real five-file Kafka replay: **534 real retraining cycles** completed through Great Expectations validation and the champion/challenger comparison with zero aborts, and the PSI-specific drift trigger (independent of the `drift_share` gate) was confirmed firing against real Tuesday attack traffic. No promotion was observed during that run: training is deterministic (`random_state=42`) against a fixed dataset, so each challenger reproduces the champion's PR-AUC exactly and correctly fails the `0.005` improvement margin. The promotion path executes and evaluates; it was not observed to fire.

---

## Stack

| Layer | Tool | Why |
|---|---|---|
| Streaming | Kafka (confluent-kafka) | At-least-once delivery, manual offset commit, DLQ |
| Feature store | Feast 0.40 + Redis 7 | Training-serving consistency, sub-ms online lookup |
| Data validation | Great Expectations | Abort retraining on bad data before it reaches the model |
| Model | LightGBM 4.x | Best tabular performance |
| Class imbalance | `scale_pos_weight` | Weight the minority attack class without resampling artifacts |
| Metric / threshold | PR-AUC + recall-calibrated threshold | Robust under extreme imbalance; recall floor for attack capture |
| Experiment tracking | MLflow 2.12 | Champion/challenger alias promotion |
| Orchestration | Prefect 2.x | 9-step retraining flow with GE validation gate |
| Drift detection | Evidently AI 0.4.30 | Drift test on features, 500-event minimum window guard |
| Serving | FastAPI + Uvicorn | Async, lifespan model hot-reload, background task logging |
| Explainability | SHAP TreeExplainer | Analyst triage and audit in network security operations |
| Monitoring | Prometheus + Grafana | Industry-standard ops stack; two provisioned dashboards |
| CI | GitHub Actions | ruff + mypy + pytest on every push |

---

## Drift Detection and Self-Healing

CICIDS2017 produces genuine distributional drift without synthetic injection: Monday is benign-only and serves as the reference distribution, while the attack-laden Tuesday-Friday captures shift the flow feature distributions (throughput, SYN counts, directional asymmetry) when replayed. The Evidently reference dataset is sampled from benign Monday traffic by `scripts/generate_reference_dataset.py`.

**Self-healing cycle:**
1. `StreamingDriftDetector` accumulates 500 events, runs an Evidently report, and computes
   two independent drift signals: `drift_share` (Evidently's `DatasetDriftMetric`) and,
   on windows where that gate doesn't already fire, a PSI score
   (`compute_psi_scores`) against the benign Monday reference
2. On a `drift_share` breach: `alert_handler.py` triggers `retraining_flow`. On a PSI
   breach (`max_psi > PSI_SIGNIFICANT_THRESHOLD`): `check_drift_and_trigger_retraining`
   triggers it independently — confirmed firing on real Tuesday attack traffic in a live
   run, distinct from and not gated by `drift_share`
3. 9-step Prefect flow: GE validation gate -> train challenger -> compare PR-AUC -> promote
4. API detects the champion version change and hot-reloads without restart

---

## Project Structure

```
driftguard/
├── src/
│   ├── config.py                     # pydantic-settings: env vars, paths, feature list, CICIDS column map
│   ├── schemas/
│   │   └── network_flow_event.avsc   # NetworkFlowEvent Avro contract (21 fields)
│   ├── data/
│   │   └── preprocess.py             # Load CICIDS2017, compute features, temporal split
│   ├── features/
│   │   ├── engineering.py            # Single source of truth: FEATURE_COLS + compute_features
│   │   ├── feature_repo/             # Feast: src_ip entity, network_flow_features (TTL 1h)
│   │   └── materializer.py           # Push offline features to Redis online store
│   ├── validation/
│   │   └── expectations.py           # GE suite: null-rate and row-count bounds on features
│   ├── training/
│   │   ├── train.py                  # LightGBM, MLflow, champion/challenger (PR-AUC margin)
│   │   ├── evaluate.py               # PR-AUC (primary), ROC-AUC, precision, recall
│   │   └── threshold.py              # calibrate_threshold to recall target
│   ├── serving/
│   │   ├── main.py                   # FastAPI lifespan, Prometheus, hot-reload loop
│   │   ├── routes.py                 # /predict, /predict/explain, /health
│   │   ├── schemas.py                # Pydantic request/response models
│   │   ├── model_loader.py           # Load champion model + threshold
│   │   └── explainer.py              # SHAP TreeExplainer, top-5 contributions
│   ├── producer/
│   │   └── flow_producer.py          # Chronological CICIDS2017 replay, re-indexed timestamps
│   ├── consumer/
│   │   ├── flow_consumer.py          # Validate, infer, log, drift-detect, commit offset
│   │   ├── schemas.py                # Pydantic NetworkFlowEvent
│   │   └── dlq_handler.py            # Route malformed messages to network_flows.dlq
│   ├── monitoring/
│   │   ├── drift_detector.py         # StreamingDriftDetector with 500-event guard
│   │   └── alert_handler.py          # Trigger retraining on drift breach
│   └── orchestration/flows/
│       └── retraining_flow.py        # 9-step Prefect flow
├── scripts/
│   └── generate_reference_dataset.py # Sample benign Monday traffic for the Evidently baseline
├── monitoring/grafana/
│   ├── dashboards/infra.json         # Request rate, latency, error rate
│   └── dashboards/ml_health.json     # Drift score, PR-AUC, DLQ rate, prediction distribution
├── benchmark/locustfile.py           # Locust load test for /predict
├── tests/                            # pytest unit tests
├── docker-compose.yml                # Kafka, Zookeeper, Redis, API, Prometheus, Grafana
└── Dockerfile.api
```

---

## Quick Start

### Prerequisites

- Python 3.11
- Docker Desktop (for full stack)
- CICIDS2017 dataset

### 1. Setup

```bash
conda create -n driftguard python=3.11 -y
conda activate driftguard
pip install -r requirements.txt
```

### 2. Download dataset

Download the five CICIDS2017 CSV files from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) into `data/raw/cicids2017/`.

### 3. Run the pipeline

```bash
# Preprocess: compute the ten features, temporal split
python -m src.data.preprocess

# Generate the Evidently reference distribution (benign Monday traffic)
python scripts/generate_reference_dataset.py

# Feast: register feature views and materialize to Redis
cd src/features/feature_repo && feast apply && cd ../../..
python -m src.features.materializer

# Train: LightGBM, PR-AUC, recall-calibrated threshold, MLflow, champion promotion
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
| Network Anomaly API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### 5. Stream flows and observe drift

```bash
# Replay CICIDS2017 flows chronologically (benign Monday -> attack days)
python -m src.producer.flow_producer

# Consumer: validate, infer, log, detect drift (separate terminal)
python -m src.consumer.flow_consumer
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

# Predict anomaly probability for a network flow
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "flow_duration": 100000.0,
    "flow_bytes_per_sec": 5000.0,
    "flow_packets_per_sec": 50.0,
    "total_fwd_packets": 10.0,
    "total_bwd_packets": 8.0,
    "packet_length_mean": 120.0,
    "packet_length_std": 30.0,
    "flow_iat_mean": 2000.0,
    "syn_flag_count": 1.0,
    "src_ip": "192.168.10.5",
    "flow_id": "192.168.10.5-52.6.13.28-49158-443-6"
  }'

# Predict with SHAP explanation (top-5 feature contributions)
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{ ...same payload... }'
```

**Example `/predict` response:**
```json
{
  "event_id": "192.168.10.5-52.6.13.28-49158-443-6",
  "anomaly_score": 0.031,
  "is_anomaly": false,
  "threshold": 0.42,
  "model_version": "20260624-143021",
  "timestamp": "2026-06-24T14:30:21.432Z"
}
```

**Example `/predict/explain` response:**
```json
{
  "anomaly_score": 0.031,
  "is_anomaly": false,
  "threshold": 0.42,
  "top_features": [
    {"feature": "syn_flag_count",       "shap_value":  0.412},
    {"feature": "fwd_bwd_packet_ratio",  "shap_value": -0.183},
    {"feature": "flow_bytes_per_sec",    "shap_value":  0.092},
    {"feature": "flow_packets_per_sec",  "shap_value": -0.071},
    {"feature": "packet_length_std",     "shap_value": -0.044}
  ]
}
```

---

## Tests

```bash
# CI (no dataset required)
pytest tests/ -m "not requires_data" -v

# Full suite
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

**PR-AUC over F2.** Under the severe class imbalance of network anomaly detection, threshold-specific Fbeta is sensitive to the operating point and misleading. PR-AUC summarizes performance across all thresholds and is the standard metric for imbalanced anomaly detection.

**Recall-calibrated serving threshold.** The decision threshold is calibrated after training to achieve recall >= 0.95 on the validation split, maximizing precision subject to that floor. A missed attack is more costly than a false alarm.

**engineering.py as single source of truth.** All feature transforms live in `src/features/engineering.py` (FEATURE_COLS + compute_features) and are called identically at training, consuming, and serving time. Any divergence causes training-serving skew, the most common source of production ML degradation.

**Benign-only reference distribution.** The Evidently reference is sampled exclusively from benign Monday traffic, giving a clean baseline for drift computation against attack-laden later days.

**500-event drift window guard.** Evidently's statistical tests are unreliable on N < 500. `StreamingDriftDetector` accumulates events silently below this threshold and logs a warning rather than running a spurious report.

**champion_metrics.json over MLflow API.** The retraining flow reads `models/champion_metrics.json` for the champion/challenger PR-AUC comparison rather than calling the MLflow tracking server, eliminating a network dependency from the critical path.

---

## Resume Bullets

- Built a real-time network anomaly detection pipeline on real CICIDS2017 traffic processing Kafka event streams through a Feast feature store (Redis online store) into a LightGBM classifier achieving **PR-AUC 0.1625** with a recall-calibrated threshold reaching **95.6% recall / 17.2% precision**, and FastAPI serving with SHAP explanations
- Implemented a self-healing retraining loop using Prefect: a Great Expectations validation gate, Evidently AI windowed drift detection (500-event guard, both `drift_share` and an independent PSI-based signal against a benign baseline), and automated MLflow champion/challenger promotion gated on PR-AUC improvement — verified live against a real five-file Kafka replay with **534 retraining cycles reaching the champion/challenger comparison with zero validation aborts**, and the PSI trigger confirmed firing on real attack traffic independent of the `drift_share` gate
- Deployed a full observability stack with Prometheus and Grafana (two provisioned dashboards), custom metrics for drift score, DLQ rate, and champion PR-AUC; at-least-once Kafka delivery with a dead letter queue for malformed events, idempotent consumer dedup, and a Confluent Schema Registry-backed Avro contract, all confirmed healthy under a 2+ hour live Docker Compose run

---

## License

MIT
