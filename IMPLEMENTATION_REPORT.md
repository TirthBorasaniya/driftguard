# P1 Domain Migration: Implementation Report

Migration of DriftGuard from credit-card fraud detection (Sparkov transactions) to
network telemetry anomaly detection on CICIDS2017 network flow data. The pipeline
architecture (Kafka, Feast, FastAPI, Prometheus/Grafana, Prefect, MLflow, Great
Expectations, Evidently) is unchanged; only domain-specific components were migrated.

## Dataset
- Name: CICIDS2017 (Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset 2017)
- Record count: 2,830,540 network flow records
- Source URL: https://www.unb.ca/cic/datasets/ids-2017.html
- Files used (in replay order):
  1. Monday-WorkingHours.pcap_ISCX.csv (benign only — drift baseline / reference)
  2. Tuesday-WorkingHours.pcap_ISCX.csv
  3. Wednesday-workingHours.pcap_ISCX.csv
  4. Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  5. Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
- Label distribution (class counts): Binary target `label_binary` = `0` for `BENIGN`,
  `1` for any of the 14 attack categories (DoS, DDoS, PortScan, Brute Force, Web
  Attacks, Infiltration, Bot, Heartbleed). Benign is the large majority class.
  Exact per-class counts are computed from the downloaded CSVs at preprocessing time
  (`src/data/preprocess.py` prints the attack rate per split); the raw CICIDS2017
  files are not present in this environment, so concrete counts are produced by
  running `python -m src.data.preprocess` after download.

## Feature Engineering
Source of truth: `src/features/engineering.py` (`FEATURE_COLS`, `compute_features`,
`compute_features_batch`). All ten features are numeric; there are no categorical
model inputs (`CATEGORICAL_COLS = []`).

1. `flow_duration`
   - Source: pass-through from event
   - Computation: total flow duration in microseconds, taken directly from the flow record
   - Rationale: long-lived or abnormally short flows distinguish scans and floods from normal sessions
2. `flow_bytes_per_sec`
   - Source: pass-through
   - Computation: bytes transferred per second over the flow lifetime
   - Rationale: volumetric attacks (DDoS) drive byte throughput far outside the benign range
3. `flow_packets_per_sec`
   - Source: pass-through
   - Computation: packets per second over the flow lifetime
   - Rationale: high packet rates characterize flooding and scanning
4. `total_fwd_packets`
   - Source: pass-through
   - Computation: total forward-direction packet count
   - Rationale: request-heavy attack flows skew forward packet volume
5. `total_bwd_packets`
   - Source: pass-through
   - Computation: total backward-direction packet count
   - Rationale: missing or minimal responses indicate one-sided attack traffic
6. `packet_length_mean`
   - Source: pass-through
   - Computation: mean packet payload length across the flow
   - Rationale: attack payloads cluster at atypical sizes versus benign sessions
7. `packet_length_std`
   - Source: pass-through
   - Computation: standard deviation of packet payload lengths
   - Rationale: uniform packet sizes (low variance) are typical of automated attack tooling
8. `flow_iat_mean`
   - Source: pass-through
   - Computation: mean inter-arrival time between packets in the flow
   - Rationale: machine-generated traffic has regular timing distinct from human-driven flows
9. `fwd_bwd_packet_ratio`
   - Source: engineered from event fields
   - Computation: `total_fwd_packets / (total_bwd_packets + EPSILON)` where `EPSILON = 1e-9`
   - Rationale: captures directional asymmetry; attacks frequently produce highly asymmetric flows versus benign bidirectional sessions
10. `syn_flag_count`
    - Source: pass-through
    - Computation: count of TCP SYN flags in the flow
    - Rationale: elevated SYN counts are a strong signal of port scans and SYN floods

## Event Schema
- Schema file: `src/schemas/network_flow_event.avsc` (Avro record `NetworkFlowEvent`,
  namespace `com.p1.pipeline`). The operative runtime schema is the Pydantic
  `NetworkFlowEvent` model in `src/consumer/schemas.py`, which mirrors the Avro contract.
- Entity key: `src_ip`
- Field count: 21 (`event_id`, `flow_id`, `timestamp_utc`, `src_ip`, `dst_ip`,
  `src_port`, `dst_port`, `protocol`, `flow_duration`, `flow_bytes_per_sec`,
  `flow_packets_per_sec`, `total_fwd_packets`, `total_bwd_packets`,
  `total_length_fwd_packets`, `total_length_bwd_packets`, `packet_length_mean`,
  `packet_length_std`, `flow_iat_mean`, `syn_flag_count`, `label`, `label_binary`)
- Avro-compatible: yes (validated as well-formed JSON; one record type, primitive field types)

## Kafka
- Primary topic: `network_flows`
- DLQ topic: `network_flows.dlq`
- Replay rate (events per second): 500 (`REPLAY_RATE_EPS` in `src/producer/flow_producer.py`, CLI-configurable via `--rate`)
- Timestamp strategy: raw CICIDS2017 timestamps are parsed with format
  `%d/%m/%Y %I:%M:%S %p`, sorted ascending, then re-indexed to current wall-clock time
  while preserving the original inter-event spacing (`timestamp_utc` = ms since epoch,
  anchored so the earliest record maps to "now"). A synthetic UUID `event_id` is
  generated per record for downstream deduplication.

## Feast
- Entity: `src_ip` (`network_entity` in `src/features/feature_repo/entities.py`)
- Feature view name: `network_flow_features`
- Feature count: 10 (one Float32 field per `FEATURE_COLS` entry)
- TTL: `timedelta(hours=1)`
- Offline store type: `file` (Parquet, backed by `data/processed/network_flow_features.parquet`); online store: Redis

## Operating Point
- Metric replacing F2: PR-AUC (average precision), `PROMOTION_METRIC = "pr_auc"`
- Justification: CICIDS2017 has severe class imbalance (benign is the large majority).
  Threshold-specific metrics such as Fbeta are sensitive to the chosen operating point
  and misleading under extreme imbalance. PR-AUC summarizes performance across all
  thresholds without requiring threshold selection during training and is the standard
  metric for imbalanced anomaly detection.
- Serving threshold calibration target: recall >= 0.95 (`SERVING_RECALL_TARGET = 0.95`).
  `calibrate_threshold` (`src/training/threshold.py`) returns the highest threshold whose
  recall meets the floor, maximizing precision subject to catching >= 95% of attacks.
  Calibration runs on the validation split after training, not during.
- Promotion margin: `PROMOTION_PRAUC_MARGIN = 0.005` (challenger must exceed champion
  PR-AUC by more than this margin; sourced from `settings.champion_improvement_threshold`).

## MLflow
- Experiment name: `network_anomaly_detection`
- Promotion metric: `pr_auc` (alias-based champion/challenger promotion unchanged; model registry name `network-anomaly-detector`)

## Files Modified
- `src/config.py` — Kafka topics, group id, MLflow experiment/model name, PR-AUC margin, feature definitions (`FEATURE_COLS` imported from engineering, `CATEGORICAL_COLS=[]`), `TARGET_COL=label_binary`, `ENTITY_COL=src_ip`, CICIDS paths and `CICIDS_COLUMN_MAP`
- `src/features/engineering.py` — replaced fraud transforms with `FEATURE_COLS`, `EPSILON`, `compute_features`, `compute_features_batch`
- `src/consumer/schemas.py` — `TransactionEvent` -> `NetworkFlowEvent` (21 fields)
- `src/consumer/dlq_handler.py` — docstring updated to `network_flows.dlq`
- `src/features/feature_repo/entities.py` — `cc_num` entity -> `src_ip` `network_entity`
- `src/features/feature_repo/feature_views.py` — single `network_flow_features` view, 10 features, TTL 1h
- `src/features/feature_repo/data_sources.py` — `network_flow_source` -> `network_flow_features.parquet`
- `src/features/feature_repo/feature_store.yaml` — project renamed to `network_anomaly_pipeline`
- `src/features/materializer.py` — import ordering (no domain logic change)
- `src/data/preprocess.py` — rewritten for CICIDS: load files, `compute_features_batch`, temporal split (train/test/stream), per-`src_ip` Feast source table
- `src/data/encoders.py` — docstring de-fraud-ed (SafeLabelEncoder retained as a utility)
- `src/validation/expectations.py` — bounded suite: `NUMERICAL_FEATURE_COLS`, `SUITE_NAME="network_flow_feature_suite"`, `validate_batch`, `MAX_NULL_RATE=0.02`, `MIN_ROW_COUNT=10_000`, `MAX_ROW_COUNT=5_000_000` (kept `run_validation_suite` wrapper for the Prefect flow)
- `src/monitoring/drift_detector.py` — imports `FEATURE_COLS` from engineering, all-numeric `ColumnMapping`, `REFERENCE_DATA_PATH` -> `data/reference/reference_network_flows.parquet`
- `src/monitoring/alert_handler.py` — implicit-Optional type fix
- `src/training/train.py` — PR-AUC primary metric, `calibrate_threshold`, `PROMOTION_METRIC`/`PROMOTION_PRAUC_MARGIN`/`SERVING_RECALL_TARGET` constants, de-fraud-ed prints
- `src/training/evaluate.py` — `pr_auc` primary, dropped `f2_score`
- `src/training/threshold.py` — `find_f2_threshold` -> `calibrate_threshold` (recall-targeted)
- `src/orchestration/flows/retraining_flow.py` — evaluate task uses `calibrate_threshold` + PR-AUC, removed unused imports
- `src/serving/schemas.py` — `TransactionRequest` -> `NetworkFlowRequest`; response fields `event_id`/`anomaly_score`/`is_anomaly`
- `src/serving/routes.py` — `build_feature_vector` uses `compute_features` (no encoders), anomaly-domain fields, predictions table columns
- `src/serving/main.py` — predictions table schema (`event_id`/`anomaly_score`/`is_anomaly`), app title/description
- `src/serving/model_loader.py` — docstring (calibrated threshold)
- `src/serving/explainer.py` — docstring + typing fix
- `src/{data,producer,training,validation}/__init__.py` — docstrings de-fraud-ed
- `monitoring/prometheus.yml`, `monitoring/grafana/.../provider.yml`, `monitoring/grafana/dashboards/{infra,ml_health}.json`, `monitoring/dashboards/model_health.json` — domain labels/tags/uids and the `predicted_class` query value (`fraud` -> `anomaly`); panel layout, scrape config, datasource wiring unchanged
- `benchmark/locustfile.py` — network flow payload, `NetworkFlowUser`
- `Makefile` — `flow_producer`/`flow_consumer` targets, `reference` target, removed `producer-drift`
- `render.yaml` — service name `driftguard-network-anomaly-api`
- `.env.example` — network topics, experiment/model name, PR-AUC margin
- `README.md` — domain description, dataset, features, operating point, commands, structure, resume bullets (architecture topology and infrastructure stack preserved)
- `tests/conftest.py`, `tests/test_config.py`, `tests/test_features.py`, `tests/test_validation.py`, `tests/test_preprocessing.py`, `tests/test_data.py`, `tests/test_serving.py`, `tests/test_api.py`, `tests/test_retraining_flow.py` — rewritten for the network domain

## Files Created
- `src/schemas/network_flow_event.avsc` — NetworkFlowEvent Avro contract (21 fields)
- `src/producer/flow_producer.py` — CICIDS2017 replay producer (`load_cicids_file`, `reindex_timestamps`, `build_flow_event`, `replay_dataset`)
- `src/consumer/flow_consumer.py` — network flow consumer (validate, compute features, infer, log, drift-detect, manual offset commit)
- `scripts/generate_reference_dataset.py` — `generate_reference_dataset` sampling benign Monday traffic for the Evidently baseline
- `src/validation/checkpoints/network_flow_checkpoint.json` — checkpoint config for the network flow suite
- `tests/test_producer.py` — `reindex_timestamps` and `build_flow_event` tests
- `tests/test_threshold.py` — `calibrate_threshold` recall-target tests
- `tests/test_feast.py` — feature view tests (skipped when `feast` is unavailable)
- `models/.gitkeep` — keep the model artifact directory after removing stale fraud artifacts

## Files Renamed
- `src/producer/transaction_producer.py` (repo: `kafka_producer.py`) -> `src/producer/flow_producer.py`
- `src/consumer/transaction_consumer.py` (repo: `kafka_consumer.py`) -> `src/consumer/flow_consumer.py`
- `src/schemas/transaction_event.avsc` -> `src/schemas/network_flow_event.avsc` (no prior Avro existed in this repo; the file was created new)

## Files Deleted
- `src/producer/kafka_producer.py`, `src/consumer/kafka_consumer.py` (superseded by `flow_producer.py`/`flow_consumer.py`)
- `src/producer/drift_injector.py`, `data/drift_scenarios/heavy_drift.json` (fraud drift injection removed; CICIDS supplies genuine attack drift)
- `src/validation/checkpoints/fraud_checkpoint.json` (superseded by `network_flow_checkpoint.json`)
- `models/{production_model.pkl,champion_metrics.json,feature_cols.json,categorical_cols.json,threshold.json,metrics_history.csv}` (stale 23-feature fraud artifacts, incompatible with the 10-feature network model; regenerated by training on CICIDS2017)

## Technical Decisions
The migration spec's file paths assume a directory layout that does not exist in this
repository (it follows the original fraud-pipeline layout). Per the user's direction to
focus on `CLAUDE_domain_migration.md`, each step was mapped onto the real files rather
than restructuring the repo. Specific decisions:

1. Path mapping — `feature_engineering.py` -> existing `src/features/engineering.py`;
   `src/feast/` -> existing `src/features/feature_repo/`; `src/api/` -> existing
   `src/serving/`; `ge_validator.py` -> existing `src/validation/expectations.py`. The
   `src/promotion/` module does not exist, so the promotion margin lives in `config.py`
   and `src/training/train.py` (`PROMOTION_PRAUC_MARGIN`), used by the Prefect flow.
2. Event schema — the repo had no Avro and validates JSON via a Pydantic model, so the
   `.avsc` was created as a static contract artifact and the operative Pydantic model was
   replaced with `NetworkFlowEvent`. No Confluent Schema Registry was wired in (that
   infrastructure does not exist in this repo).
3. Drift injection removed — the spec defines the producer as replay-only, and CICIDS2017
   produces genuine distributional drift (benign Monday baseline vs. attack-laden
   Tuesday-Friday), so the fraud `drift_injector` and `--drift` mode were removed rather
   than rewritten. The Evidently reference is the benign Monday sample.
4. Great Expectations — the repo's GE suite was already a hand-rolled pandas
   implementation (no GE `DataContext`). `validate_batch` was implemented in the same
   style with the spec's bounded null-rate (2%) and row-count constants, keeping the suite
   runnable without a DataContext.
5. `calibrate_threshold` interpretation — returns the highest threshold whose recall meets
   the 0.95 floor (precision-maximizing while guaranteeing attack capture), which is the
   correct serving operating point and satisfies the spec's recall-target test.
6. Consumer entity — the consumer does not perform Feast online writes in this repo, so the
   `src_ip` entity change is reflected in the event schema and the feature view; the DLQ
   routing, deduplication-by-`event_id` capability, and manual offset-commit pattern were
   left intact.
7. `kafka_group_id` value changed from `fraud-consumer` to `network-flow-consumer` to remove
   the forbidden `fraud` token; the constant name is unchanged.
8. Lazy `confluent_kafka` import in `flow_producer.py` so the pure replay helpers
   (`reindex_timestamps`, `build_flow_event`) are unit-testable without the native client.
9. Stale fraud model artifacts were removed because the committed 23-feature fraud model is
   incompatible with the 10-feature network model and would error at inference; they are
   regenerated by `make data && make train` on CICIDS2017.

## Verification
- Environment note: the active interpreter is Anaconda Python 3.13 (the project targets
  3.11) and the heavy ML dependencies (lightgbm, mlflow, fastapi, evidently, feast,
  great_expectations, shap, confluent_kafka) are not installed. Installing the pinned
  `requirements.txt` here would downgrade global numpy/pandas and was not performed.
- `ruff check src/ tests/ scripts/` — clean.
- `mypy src/ --ignore-missing-imports` — clean (36 files).
- `pytest tests/` — 49 passed, 21 skipped. The 21 skips are dependency-gated via
  `pytest.importorskip` (serving: fastapi/mlflow/aiosqlite/prometheus_fastapi_instrumentator;
  retraining promotion logic: lightgbm/mlflow; Feast feature view: feast). On CI
  (Python 3.11 with `requirements.txt` installed) these dependencies are present and the
  full suite runs. All spec-required tests that do not need the absent libraries pass:
  `compute_features` returns exactly `FEATURE_COLS`; `fwd_bwd_packet_ratio == 1.0` for
  equal packet counts; `reindex_timestamps` first record within 5s of now; `build_flow_event`
  includes `event_id` and all schema fields; `validate_batch` fails on >2% null rate;
  `calibrate_threshold` achieves recall >= `SERVING_RECALL_TARGET`. The Feast feature-view
  test is implemented and runs when `feast` is installed.
- End-to-end run on real CICIDS2017 data (download -> preprocess -> train -> serve ->
  stream -> dashboards) was not performed: the raw dataset is not present in this
  environment and the heavy runtime dependencies are not installed.
