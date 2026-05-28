# CLAUDE.md — Real-Time Event-Driven ML Pipeline (P1)

---

## Strict Scope Instructions

These instructions are mandatory and override any inferred behaviour.

**Claude Code must only read, write, and execute files within this project
folder.** No exceptions.

- Do NOT read, access, reference, or modify any file outside this folder.
- Do NOT traverse parent directories (`../`, `~`, `/etc`, `/usr`, or any
  path that resolves outside the project root).
- Do NOT read shell history, global config files, other projects, or any
  system-level file.
- Do NOT infer context from files that exist on the host machine outside
  this folder. If information is not in this CLAUDE.md or in a file within
  this folder, ask for it explicitly rather than looking elsewhere.
- Do NOT run commands that produce output from outside this folder
  (e.g., `ls ~`, `cat ~/.bashrc`, `env`, `printenv`, `whoami`).

**Permitted external actions:**

- `pip install` from requirements.txt (writes to active virtualenv only)
- `git` commands scoped to this repository
- `docker compose` commands referencing docker-compose.yml in this folder
- Network requests made by the running application as part of normal operation
- Fetching public documentation URLs when explicitly asked to research something

Any action outside these boundaries requires explicit confirmation before
proceeding.

---

## Project Objective

Real-time fraud detection system demonstrating production ML engineering:
Kafka event ingestion with schema validation, Feast feature store with Redis
online store, Great Expectations data validation, LightGBM training with
MLflow experiment tracking, FastAPI inference serving with SHAP explanations,
Evidently AI drift detection, Prometheus and Grafana infrastructure monitoring,
and a Prefect retraining loop with automated champion-challenger promotion.
Target roles: MLE, MLOps, Data Engineer.

---

## Python Version

**3.11 strictly.** Do not use 3.12 or 3.13.

---

## Stack

| Layer                     | Tool                               | Version   | Notes                                   |
|---------------------------|------------------------------------|-----------|-----------------------------------------|
| Streaming                 | Kafka (confluent-kafka)            | latest    | Docker Compose service                  |
| Feature store             | Feast                              | 0.40.x    | Redis online store, Parquet offline     |
| Online store              | Redis                              | 7.x       | AOF persistence enabled                 |
| Model                     | LightGBM                           | 4.x       | Early stopping via callbacks only       |
| Experiment tracking       | MLflow                             | 2.12.x    | SQLite backend, alias-based promotion   |
| Orchestration             | Prefect                            | 2.x       | Must be < 3.0.0                         |
| Data validation           | Great Expectations                 | latest    | First step in retraining loop           |
| Drift detection           | Evidently AI                       | 0.4.30    | Pinned exactly — do not upgrade         |
| Infrastructure monitoring | Prometheus + Grafana               | latest    | Separate from Evidently ML monitoring   |
| API instrumentation       | prometheus-fastapi-instrumentator  | latest    |                                         |
| Serving                   | FastAPI + Uvicorn                  | 0.111.x   | Lifespan context manager only           |
| Explainability            | SHAP                               | latest    | TreeExplainer, top-5 contributions      |
| Event validation          | Pydantic                           | 2.x       |                                         |
| Infra                     | Docker Compose (no version:)       | latest    | Compose Specification format            |
| CI                        | GitHub Actions                     | —         | ruff, mypy, pytest on push              |
| Dataset                   | Synthetic Credit Card Transactions | Kaggle    | Brandon Harris / Sparkov simulator      |

---

## Dataset

Source: Kaggle — "Credit Card Transactions Fraud Detection Dataset"
by Brandon Harris. Generated using the Sparkov simulator, the same
simulation engine used by financial institutions for testing fraud systems.

Files: `fraudTrain.csv` and `fraudTest.csv`.
1.85 million transactions across 24 months, 1000 customers, 800 merchants.

**Entity identifier:** `cc_num` (credit card number). Single-column stable
entity identifier that appears repeatedly across transactions. Feast entity
definition is direct — no compound key construction required. This is the
primary reason this dataset is preferred over IEEE-CIS, which requires a
derived compound key from card1-card6 columns.

**Temporal split:** Sort by `trans_date_trans_time`, split at the 75th
percentile. Never random split on time-ordered fraud data — causes leakage.
The 24-month window produces genuine seasonal drift: spending patterns shift
by season, merchant category distributions evolve, and fraud patterns change
over time. This fires drift detection naturally without requiring injection.

**Fraud rate:** Approximately 0.58%. Use `scale_pos_weight=172`
(99.42 / 0.58). Do NOT use SMOTE. SMOTE degrades performance on
high-cardinality categorical features by interpolating in a space where
interpolation is semantically invalid, and introduces leakage risk if
applied before splitting.

**Named features available:**
- `cc_num` — entity identifier (Feast entity key)
- `merchant`, `category` — merchant name and spending category (categorical)
- `amt` — transaction amount
- `gender`, `city`, `state`, `zip` — cardholder demographics
- `lat`, `long` — cardholder location
- `city_pop` — population of cardholder city
- `job`, `dob` — cardholder occupation and date of birth
- `merch_lat`, `merch_long` — merchant location
- `trans_date_trans_time` — transaction timestamp (used for temporal split)
- `is_fraud` — target label

**Derived features to engineer:**
- `hour_of_day`, `day_of_week` — extracted from `trans_date_trans_time`
- `age` — derived from `dob` relative to transaction date
- `distance_km` — haversine distance between cardholder and merchant coords
- `amt_log` — log1p transform of `amt` to reduce right skew

**Drift:** The 24-month window contains natural seasonal patterns and
evolving merchant category distributions. Use temporal split at 75th
percentile for training. Stream the remaining 25% chronologically for the
live demo. The producer also supports a drift injection mode for controlled
testing: scale `amt` by a configurable multiplier, shift `category` ratios,
increase transaction concentration from a specific `state`. Document both
natural and injected drift modes in the README.

---

## Architecture Decisions

**Class imbalance:** `scale_pos_weight` only, no SMOTE.

**Prediction threshold:** Optimized at F2 operating point on Precision-Recall
curve, not default 0.5. Store threshold as JSON artifact alongside model
binary in MLflow. Load at serving time.

**Categorical encoding:** Custom encoder that maps unseen values to a
reserved bin (-1) rather than raising ValueError. LabelEncoder raises on
unseen values and will crash the API on new merchant types or new states.
Save fitted encoders to `data/encoders/`. Load at API startup. The API
accepts raw string categorical values and encodes them internally at inference.

**LightGBM categorical_feature:** Compute `cat_indices` as list of column
positions for categorical features (`category`, `gender`, `state`, `job`
after encoding). Pass `categorical_feature=cat_indices` to `model.fit()`.
Without this, LightGBM treats label-encoded integers as ordered numeric
values, which is semantically incorrect and degrades split quality.

**Feast entity:** `cc_num` is the primary entity. Single column, stable
across transactions, no derivation required.

**Feast feature views:** Define two aggregated views:
- Rolling 7-day transaction stats per `cc_num`:
  transaction count, mean amount, max amount, transaction velocity
- Rolling fraud statistics per merchant `category`:
  fraud count, total count, rolling fraud rate

Scalar transaction-level attributes alone do not justify a feature store.
The aggregations are what require consistent computation across training
and serving, and what make Feast's push API meaningful.

**Feast online store:** Redis only, not SQLite. SQLite is single-threaded
and unsafe under concurrent FastAPI requests. Feature lookup latency with
Redis is sub-millisecond.

**Great Expectations data validation:** Run a validation suite on the
training dataset as the first task in the Prefect retraining flow. Define
expectations on:
- `amt` > 0, no nulls
- `city_pop` > 0, no nulls
- `category` cardinality between 10 and 20 unique values
- `state` cardinality between 40 and 60 unique values
- `is_fraud` rate between 0.3% and 2%
- No duplicate `trans_num` values

If validation fails, abort the retraining run immediately, log the specific
failed expectations, and do not proceed to training. Never train on data
that has not passed validation. Store GE checkpoint config in
`src/validation/checkpoints/`.

**Kafka consumer:** Set `enable.auto.commit=False`. Commit offset manually
after full processing: features computed, prediction made, result logged.
This gives at-least-once delivery semantics with explicit control.

**Dead letter queue:** Validate every incoming Kafka message against the
`TransactionEvent` Pydantic model before processing. Malformed messages go
to a `transactions.dlq` topic with the validation error attached as a header.
Never silently drop malformed events.

**Drift detection window:** Minimum 500 events before computing drift report.
Below this threshold, log a warning and skip the drift computation for that
cycle. Evidently's KS test and PSI are unreliable on small windows.

**Retraining loop (9 steps, in order):**
1. Run Great Expectations validation suite on incoming training data
2. Abort if validation fails — log failed expectations, do not proceed
3. Materialize features from Feast offline store
4. Load training data with temporal split enforced
5. Train LightGBM challenger with scale_pos_weight=172
6. Evaluate challenger on held-out test set (AUC-PR, F2-score at optimized threshold)
7. Load champion metrics from artifact JSON (not MLflow API — avoids API dependency)
8. Compare AUC-PR: challenger wins if improvement > 0.01
9. Promote challenger via MLflow model alias and signal FastAPI to reload

Steps 8 and 9 are what make the loop self-healing. Without conditional
promotion, retraining is just scheduled training.

**Champion metrics storage:** At registration time, write champion metrics
(AUC-PR, F2-score, threshold) to a JSON file in the model artifact directory.
Promotion flow reads this file directly. No MLflow API dependency during
the comparison and promotion decision.

**SHAP explanations:** `/predict/explain` endpoint returns top-5 SHAP
feature contributions per prediction using `shap.TreeExplainer`. This is
a regulatory requirement in production fraud detection contexts and
demonstrates awareness of real-world ML deployment constraints.

**Infrastructure monitoring:** Add `prometheus-fastapi-instrumentator` to
the FastAPI serving layer. Exposes `/metrics` endpoint automatically.
Prometheus scrapes this endpoint on a 15-second interval. Grafana is
provisioned via `monitoring/grafana/provisioning/` with two dashboards:
- `infra.json`: request rate, p50/p95/p99 latency, HTTP error rate by code
- `ml_health.json`: prediction throughput, fraud rate over time, drift alert events

This separates concerns correctly: Evidently owns ML health (did the data
distribution change?), Prometheus and Grafana own infrastructure health
(is the API responding, at what latency, with what error rate?).

**Redis persistence:** Enable AOF in docker-compose to survive container
restarts without losing the online feature store:
```yaml
redis:
  image: redis:7
  command: redis-server --appendonly yes --appendfsync everysec
  volumes:
    - redis_data:/data
```

---

## Breaking API Changes

**LightGBM 4.x:** `early_stopping_rounds` removed from `fit()`. Use:
```python
callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
```

**MLflow 2.9+:** Stage-based promotion deprecated. Use model aliases:
```python
client.set_registered_model_alias(model_name, "champion", version)
client.set_registered_model_alias(model_name, "challenger", version)
```
Load by alias:
```python
mlflow.lightgbm.load_model(f"models:/{model_name}@champion")
```

**FastAPI:** `@app.on_event("startup")` deprecated. Use lifespan:
```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: load model, encoders, threshold
    yield
    # shutdown: release resources

app = FastAPI(lifespan=lifespan)
```

**Docker Compose:** `depends_on: condition: service_healthy` requires
Compose Specification format. Remove the `version:` field at the top of
docker-compose.yml entirely.

**Evidently AI:** Pinned at 0.4.30. The 0.7+ API is completely different.
Do not upgrade. Install explicitly:
```
evidently==0.4.30
```

**aiosqlite WAL mode:** Enable WAL explicitly after connection to prevent
locking under concurrent FastAPI requests:
```python
async with aiosqlite.connect("predictions.db") as db:
    await db.execute("PRAGMA journal_mode=WAL")
```

---

## Directory Structure

```
realtime-fraud-pipeline/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── docker-compose.yml
│
├── data/
│   ├── raw/                          # fraudTrain.csv, fraudTest.csv (gitignored)
│   ├── processed/                    # train.parquet, test.parquet after preprocessing
│   ├── reference/                    # Evidently AI reference dataset (training distribution)
│   ├── encoders/                     # fitted custom encoder objects (.pkl)
│   └── drift_scenarios/              # drift injection config files (heavy_drift.json)
│
├── monitoring/
│   └── grafana/
│       └── provisioning/
│           ├── datasources/
│           │   └── prometheus.yml    # Prometheus datasource config
│           └── dashboards/
│               ├── infra.json        # request rate, latency, error rate panels
│               └── ml_health.json    # fraud rate, drift events, prediction throughput
│
├── src/
│   ├── config.py                     # pydantic-settings: all env vars and constants
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py             # feature engineering, derived features, custom encoder, temporal split
│   │   └── encoders.py               # custom encoder with unseen-category fallback bin (-1)
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── expectations.py           # GE suite: amt, city_pop, category cardinality, fraud rate, trans_num uniqueness
│   │   └── checkpoints/              # GE checkpoint configs (committed to repo)
│   │
│   ├── producer/
│   │   ├── __init__.py
│   │   ├── kafka_producer.py         # chronological replay of stream.parquet as Kafka events
│   │   └── drift_injector.py         # drift mode: amt scaling, category shift, state concentration
│   │
│   ├── consumer/
│   │   ├── __init__.py
│   │   ├── kafka_consumer.py         # manual offset commit, at-least-once delivery
│   │   ├── schemas.py                # Pydantic TransactionEvent model
│   │   └── dlq_handler.py            # dead letter queue publisher for malformed events
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_repo/
│   │   │   ├── feature_store.yaml    # Redis online store, Parquet offline store
│   │   │   ├── entities.py           # cc_num entity definition
│   │   │   ├── feature_views.py      # rolling 7-day stats per cc_num, fraud rate per category
│   │   │   └── data_sources.py       # FileSource pointing to data/processed/
│   │   ├── engineering.py            # feature transformation logic (single source of truth)
│   │   └── materializer.py           # offline store materialization script
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                  # LightGBM training, MLflow logging, champion alias promotion
│   │   ├── evaluate.py               # AUC-PR, F2-score, threshold optimization on PRC
│   │   └── threshold.py              # F2 operating point computation, JSON artifact write
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI lifespan, router mount, Prometheus instrumentation
│   │   ├── routes.py                 # /predict, /predict/explain, /health, /metrics
│   │   ├── schemas.py                # Pydantic request/response models
│   │   ├── model_loader.py           # MLflow model + threshold loading by @champion alias
│   │   └── explainer.py              # SHAP TreeExplainer wrapper, top-5 contributions
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── drift_detector.py         # Evidently windowed drift reports, 500-event minimum guard
│   │   └── alert_handler.py          # Prefect flow trigger on drift threshold breach
│   │
│   └── orchestration/
│       ├── __init__.py
│       └── flows/
│           ├── retraining_flow.py    # 9-step loop: GE validation first, conditional promotion last
│           └── materialization_flow.py
│
├── tests/
│   ├── test_data.py
│   ├── test_validation.py
│   ├── test_features.py
│   ├── test_serving.py
│   └── test_retraining_flow.py
│
└── .github/
    └── workflows/
        └── ci.yml                    # ruff lint, mypy, pytest on push
```

---

## Code Style

- NumPy-style docstrings on all functions
- `o_` prefix on boolean flag parameters (e.g., `o_run_from_scratch`)
- `_dict` suffix on key-value mappings (e.g., `feature_dict`)
- `_list` suffix on lists of items per group (e.g., `cat_cols_list`)
- No emojis, no em dashes
- Formal technical language throughout
- Imports in three groups separated by blank line: stdlib, third-party, local
- Module-level docstring on every file (one to two sentences)
- f-strings for all string formatting
- Specific exception types in try/except, never bare except

---

## Implementation Checklist

- [x] Part 1: Project scaffold — pyproject.toml, requirements.txt, .env.example, config.py
- [x] Part 2: Data pipeline — load fraudTrain.csv, engineer derived features, custom encoder, temporal split at 75th percentile
- [x] Part 3: Great Expectations suite — expectations on amt, city_pop, category cardinality, state cardinality, is_fraud rate, trans_num uniqueness
- [x] Part 4: Feast feature store — feature_store.yaml, cc_num entity, two feature views (rolling cc_num stats + category fraud rate), feast apply
- [x] Part 5: Feature materialization — offline store population from processed data
- [x] Part 6: LightGBM training — scale_pos_weight=172, categorical_feature indices, early stopping via callbacks, MLflow logging, champion alias
- [x] Part 7: Threshold optimization — F2 on PRC, store as artifact JSON alongside model binary
- [x] Part 8: FastAPI serving — lifespan, prometheus-fastapi-instrumentator, /predict, /predict/explain (SHAP top-5), /health
- [x] Part 9: Kafka producer — chronological replay of stream.parquet, drift injection mode
- [x] Part 10: Kafka consumer — Pydantic TransactionEvent validation, DLQ for malformed events, manual offset commit, aiosqlite WAL logging
- [x] Part 11: Evidently drift detector — 500-event minimum window guard, StreamingDriftDetector class
- [x] Part 12: Prefect retraining flow — GE validation as step 1 (abort on failure), 9-step loop, conditional MLflow alias promotion
- [x] Part 13: Prometheus + Grafana — two provisioned dashboards (infra.json + ml_health.json)
- [x] Part 14: Docker Compose — kafka, zookeeper, redis (AOF), api, prometheus, grafana with health checks
- [x] Part 15: GitHub Actions CI — ruff, mypy, pytest on push
- [x] Part 16: Latency benchmark — locust locustfile.py targeting /predict
- [ ] Part 17: Download fraudTrain.csv + fraudTest.csv, run pipeline end-to-end, verify all dashboards

Update checkboxes as parts are completed.

---

## Common Commands

```bash
# Environment setup
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Data pipeline
python -m src.data.preprocess

# Run Great Expectations validation standalone
python -m src.validation.expectations

# Feast setup
cd src/features/feature_repo && feast apply && cd ../../..
python -m src.features.materializer

# Training
python -m src.training.train

# Serving
uvicorn src.serving.main:app --reload --host 0.0.0.0 --port 8000

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# Kafka producer (normal mode)
python -m src.producer.kafka_producer

# Kafka producer (drift injection mode)
python -m src.producer.kafka_producer --drift

# Kafka consumer
python -m src.consumer.kafka_consumer

# Drift detection
python -m src.monitoring.drift_detector

# Prefect retraining flow
python -m src.orchestration.flows.retraining_flow

# Full stack
docker compose up --build

# Tests
pytest tests/ -v

# Latency benchmark
locust -f locustfile.py --host=http://localhost:8000

# Verify Prometheus metrics endpoint
curl http://localhost:8000/metrics

# Verify Great Expectations
python -c "import great_expectations as ge; print(ge.__version__)"

# Verify Redis AOF persistence
docker exec -it redis redis-cli CONFIG GET appendonly

# Verify Evidently version is exactly 0.4.30
python -c "import evidently; print(evidently.__version__)"
```
