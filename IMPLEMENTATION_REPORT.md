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

---

# Infra Improvements (P1_COMBINED_SESSION.md, Phases 2-4)

Each section documents the audit finding, what was implemented, and any deviation from
the spec with justification. Sections are added as each improvement is completed.

## 1. Dead Letter Queue Hardening (Tier 1)
- Audit finding: PARTIAL. `send_to_dlq` existed and published to `network_flows.dlq`
  with a header, but the header key was `"error"`, not the spec's `"validation_error"`,
  and there was no `DLQ_ERROR_HEADER_KEY` constant.
- Implemented: `src/consumer/dlq_handler.py` renamed `send_to_dlq` -> `route_to_dlq`
  matching the spec signature `(producer, raw_message, validation_error, dlq_topic)`;
  added `DLQ_TOPIC` and `DLQ_ERROR_HEADER_KEY = "validation_error"` constants; the
  header now uses `DLQ_ERROR_HEADER_KEY`. Updated the one call site in
  `src/consumer/flow_consumer.py`.
- Tests: `tests/test_dlq.py` — asserts a routed message lands on `DLQ_TOPIC` with the
  `validation_error` header set to the encoded error string. Skipped locally via
  `pytest.importorskip("confluent_kafka")` since the native client is not installed in
  this environment; runs on CI.
- Deviation: none.

## 2. PSI Drift Detection (Tier 1)
- Audit finding: PARTIAL. `alert_handler.py` already triggered `retraining_flow()` on
  drift (not just logging), but the gate was Evidently's `drift_share`/
  `DatasetDriftMetric`, a different signal than PSI. No PSI-specific constants or
  trigger function existed.
- Implemented: `src/monitoring/drift_detector.py` adds `PSI_MINOR_THRESHOLD = 0.1`,
  `PSI_SIGNIFICANT_THRESHOLD = 0.25`, `DRIFT_MIN_WINDOW = 500`, a `compute_psi_scores`
  function using Evidently's `ColumnDriftMetric(stattest="psi")` per feature column, and
  `check_drift_and_trigger_retraining(psi_score, significant_threshold,
  retraining_flow_fn)` exactly matching the spec signature. `StreamingDriftDetector.
  add_event` now runs the PSI computation on every window (after the existing
  `drift_share` check finds no dataset-level drift) and calls
  `check_drift_and_trigger_retraining` against the Prefect `retraining_flow`, so the two
  signals coexist rather than one replacing the other, per the user's correction.
- Tests: `tests/test_psi_trigger.py` — asserts the flow function is called when
  `psi_score` exceeds `PSI_SIGNIFICANT_THRESHOLD` and is not called below
  `PSI_MINOR_THRESHOLD`. Skipped locally via `pytest.importorskip("evidently")`; runs on
  CI.
- Deviation: the existing `drift_share` gate was left in place per the user's explicit
  instruction not to rename or remove it; the PSI path is a genuinely separate signal
  computed via Evidently's per-column PSI stattest, not a relabeling of `drift_share`.

## 3. Latency SLOs (Tier 1)
- Audit finding: MISSING. No alert rule file, no `rule_files` wiring in
  `monitoring/prometheus.yml`, no SLO panel in `infra.json`.
- Implemented: created `monitoring/prometheus/alerts.yml` with the
  `PredictLatencyP95Breach` (>150ms, warning) and `PredictLatencyP99Breach` (>300ms,
  critical) rules from the spec. Wired `rule_files: ["prometheus/alerts.yml"]` into
  `monitoring/prometheus.yml`, and mounted `./monitoring/prometheus` into the Prometheus
  container in `docker-compose.yml` so the relative rule path resolves. Added an
  "/predict Latency SLO Burn" panel to `infra.json` showing p95/p99 against the 150ms/
  300ms thresholds via Grafana threshold steps.
- Tests: `tests/test_latency_slo.py` — asserts `alerts.yml` parses as valid YAML,
  contains the `latency_slo` group with both alert names, and each rule's `expr`
  references `http_request_duration_seconds_bucket` and `handler="/predict"`.
- Deviation: fixed a YAML structural bug in the spec's own example (the
  `PredictLatencyP99Breach` rule had `summary` nested under `labels` instead of
  `annotations`) since a literal copy would not parse correctly.

## 4. Bounded Great Expectations Rules (Tier 1)
- Audit finding: MISSING. `expectations.py` only checked null rate and row count.
- Implemented: `src/validation/expectations.py` adds the `FEATURE_BOUNDS` dict (all ten
  features bounded at `(0.0, None)` per the spec), a `_check_bounds` helper, and
  `add_bounded_expectations(ge_suite, feature_bounds_dict)` matching the spec signature.
  Since this repo's validation suite is a hand-rolled dict/list config (no real GE
  `ExpectationSuite` object — noted in the Phase 1 report), `ge_suite` is the same
  checkpoint dict structure used by `save_checkpoint_config`. `validate_batch` now runs
  bound checks on every `FEATURE_BOUNDS` column in addition to null-rate and row-count,
  and `save_checkpoint_config` calls `add_bounded_expectations` so the persisted
  checkpoint documents the bounds.
- Tests: `tests/test_validation.py` — `test_negative_value_in_bounded_column_fails`
  asserts a synthetic batch with a negative `flow_bytes_per_sec` value fails validation;
  `test_add_bounded_expectations_extends_suite` asserts all bounded columns are added to
  a suite dict.
- Deviation: `add_bounded_expectations` operates on a plain dict rather than a real GE
  `ExpectationSuite` object, consistent with the rest of this repo's hand-rolled
  validation suite (documented as a Phase 1 technical decision).

## 5. Idempotent Consumer with Redis Dedup (Tier 2)
- Audit finding: MISSING. No Redis usage in `flow_consumer.py` at all.
- Implemented: new `src/consumer/dedup.py` with `DEDUP_KEY_PREFIX = "processed_event:"`,
  `DEDUP_TTL_SECONDS = 3600`, and `is_duplicate_event(redis_client, event_id,
  dedup_key_prefix, dedup_ttl_seconds)` using `redis-py`'s `SET key val EX ttl NX` for
  atomic check-and-mark. Wired into `src/consumer/flow_consumer.py`'s main loop as step 4
  (renumbering the docstring's processing steps to 9 total), immediately after DLQ
  validation and before feature computation as specified: on a duplicate, the offset is
  committed and the message is skipped without reprocessing.
- Tests: `tests/test_dedup.py` — asserts `is_duplicate_event` returns `False` on first
  call and `True` on a second call with the same `event_id`; different `event_id`s are
  independent; the Redis key uses `DEDUP_KEY_PREFIX`. Uses an in-memory `FakeRedis`
  implementing `SET ... EX ... NX` semantics rather than a live Redis server; skipped via
  `pytest.importorskip("redis")` locally, runs on CI.
- Deviation: none.

## 6. Shadow Mode Champion-Challenger (Tier 2)
- Audit finding: MISSING. The pre-existing `healing_mode="SHADOW"` string in
  `alert_handler.py` is an unrelated concept (retrain-and-register-as-challenger-only,
  not live dual-model scoring) and was left untouched per the user's explicit
  instruction.
- Implemented: new `src/serving/shadow_mode.py` with `SHADOW_MODE_MIN_EVENTS = 500`,
  `SHADOW_DIVERGENCE_THRESHOLD = 0.05`, `score_shadow_mode(champion_model,
  challenger_model, feature_dict)` (returns both predicted probabilities, champion
  first), `log_shadow_prediction` (appends a prediction pair plus each model's binary
  decision to a CSV shadow log), and `evaluate_shadow_divergence(shadow_log_path,
  min_events, divergence_threshold)` (disagreement-rate gate for promotion). Wired
  minimally into serving: `src/serving/model_loader.py` gains `load_challenger()`
  (loads the MLflow `@challenger` alias, returns `None` if none is registered) and
  `ModelBundle.challenger_model`; `src/serving/routes.py`'s `/predict` endpoint scores
  the challenger in the background and logs the pair to `SHADOW_LOG_PATH`
  (`models/shadow_predictions.csv`) whenever a challenger is loaded, without changing
  the response served to the client (champion prediction only).
- Tests: `tests/test_shadow_mode.py` — asserts `score_shadow_mode` returns both
  predictions without raising when the models disagree; `evaluate_shadow_divergence`
  rejects below `SHADOW_MODE_MIN_EVENTS`, approves low disagreement, rejects high
  disagreement. All pass locally (no optional dependency required).
- Deviation: the live-traffic wiring uses a CSV log rather than the SQLite predictions
  database, since the spec's `evaluate_shadow_divergence` signature takes a
  `shadow_log_path` (file path), not a database handle; this keeps the promotion-gate
  read path dependency-free (no async DB driver needed in the offline evaluation
  script). The challenger-scoring code path itself (loading via MLflow, calling
  `score_shadow_mode` inside `/predict`) was not exercised end-to-end since FastAPI and
  MLflow are not installed in this environment — only the standalone functions were
  test-verified.

## 7. Confluent Schema Registry (Tier 3)
- Audit finding: MISSING. `docker-compose.yml` had no `schema-registry` service; no
  registry code anywhere in `src/`; producer/consumer serialized plain JSON despite the
  static `.avsc` file existing.
- Implemented: added a `schema-registry` service (`confluentinc/cp-schema-registry`,
  port 8081, health-checked) to `docker-compose.yml`, wired the `api` service to depend
  on it and read `SCHEMA_REGISTRY_URL`. New `src/schemas/registry.py` with
  `SCHEMA_REGISTRY_URL`, `SUBJECT_NAME = "network_flows-value"`, `register_schema`
  matching the spec signature exactly, plus `build_avro_serializer`/
  `build_avro_deserializer` factory helpers. `src/producer/flow_producer.py`'s
  `run_producer` now registers the schema and builds an `AvroSerializer` at startup,
  falling back to plain JSON (with a printed warning) if the registry is unreachable;
  `replay_dataset` takes an optional `avro_serializer` and uses it when present.
  `src/consumer/flow_consumer.py` mirrors this with `build_avro_deserializer` and a
  `SerializationError`-aware except clause. Added `fastavro` to `requirements.txt`
  (required by `confluent_kafka.schema_registry.avro`).
- Tests: `tests/test_schema_registry.py` — asserts `register_schema` returns a positive
  integer schema ID, skipped both when `confluent_kafka.schema_registry` is unavailable
  and when no live registry is reachable at `SCHEMA_REGISTRY_URL` (this test genuinely
  requires the Docker Compose `schema-registry` service running; it is not mockable
  without testing against a fake registry implementation).
- Deviation: producer/consumer fall back to JSON when the registry is unreachable
  rather than failing hard, consistent with this repo's existing resilience pattern
  (e.g. `model_loader.py`'s MLflow-then-local-file fallback) so local development
  without the full Docker Compose stack still works.

## 8. Feast Offline Materialization (Tier 3)
- Audit finding: PARTIAL. `src/features/materializer.py` existed but only wrapped
  `feast apply` and `feast materialize-incremental` (online store push); no distinct
  offline-only path, and `src/orchestration/flows/materialization_flow.py` did not
  exist as a standalone file (only an unrelated `materialization_flow()` function
  inside `retraining_flow.py` that also just calls the online materializer).
- Implemented: `src/features/materializer.py` adds `MATERIALIZATION_LOOKBACK_HOURS = 24`
  and `materialize_offline_features(feast_repo_path, lookback_hours)` matching the spec
  signature. It builds a point-in-time correct query against the offline store: reads
  the per-`src_ip` offline parquet table, takes the most recent `event_timestamp` per
  `src_ip` within the lookback window as an entity dataframe, and calls
  `FeatureStore.get_historical_features(...).to_df()` — Feast's point-in-time join
  against the offline store, which never touches the online Redis store used at serving
  time. Created `src/orchestration/flows/materialization_flow.py` as a new file with a
  `offline-materialization-flow` Prefect flow wrapping this function, kept distinctly
  named from the pre-existing `materialization_flow()` in `retraining_flow.py` (which
  refreshes the online store) to avoid a name collision or conflating the two paths.
- Tests: `tests/test_materialization.py` — asserts `materialize_offline_features`
  completes without error against a populated offline parquet fixture and returns a
  non-empty DataFrame containing `src_ip`. Skipped via `pytest.importorskip("feast")`
  locally; runs on CI.
- Deviation: `materialize_offline_features` returns the resulting DataFrame (spec's
  signature declares no return value) so both the flow and the test can inspect the
  output directly, matching the spec's own test requirement ("produces a non-empty
  output").
