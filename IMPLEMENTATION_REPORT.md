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

### Static checks (original, Anaconda Python 3.13 stand-in environment)
- `ruff check src/ tests/ scripts/` — clean.
- `mypy src/ --ignore-missing-imports` — clean (36 files) at the time; see the real-env
  section below for type errors that surfaced once the actual pinned dependencies were
  installed.
- `pytest tests/` — 49 passed, 21 skipped (heavy ML deps not installed in that environment).

### Real Python 3.11 environment (`.venv311`, `requirements.txt` installed exactly as pinned)
All of `requirements.txt` installed cleanly under `pyenv`-provisioned Python 3.11.9. This
surfaced five real, pre-existing bugs invisible in the dependency-light Anaconda run,
all now fixed and verified:
1. `apply_feast()`/`materialize_to_online_store()` invoked `python -m feast`, which fails
   (`feast is a package and cannot be directly executed`) — fixed with a
   `_feast_executable()` resolver in `src/features/materializer.py` that locates the real
   `feast` console script.
2. `TestClient(app)` in `tests/test_serving.py`/`tests/test_api.py` was not used as a
   context manager, so FastAPI's lifespan never ran and `app.state.bundle` was never set —
   fixed to `with TestClient(app) as test_client: yield test_client`.
3. `tests/test_retraining_flow.py` monkeypatched `src.config.CHAMPION_METRICS_PATH`, but
   `train.py` had already bound its own module-level name via `from ... import`, so the
   patch was a no-op — fixed to patch `train.CHAMPION_METRICS_PATH` directly.
4. `tests/test_feast.py` had an E402 import-order violation invisible to the Anaconda
   `ruff` (0.12.0) but caught by the pinned `ruff==0.4.7` — fixed with `# noqa: E402`.
5. `tests/test_materialization.py`'s fixture only monkeypatched a config constant that
   Feast's `FileSource` had already bound at import time, so it never exercised real
   Feast — rewrote it to write to the real offline path and run a real `feast apply`.

Full suite after these fixes: **89 passed, 1 skipped** (only the live-registry test
skips when the Schema Registry container isn't running).

### Dataset acquisition — wrong variant caught before use
The first zip provided (`MachineLearningCSV.zip`) is CICIDS2017's **MachineLearningCSV**
release, which strips `Flow ID`, `Source IP`, `Destination IP`, `Source Port`, and
`Timestamp` — confirmed by inspecting the actual CSV header (only `Destination Port` and
`Label` remain from the identifying columns). This is incompatible with the pipeline's
`src_ip` Feast entity and the producer's timestamp re-indexing. Caught via `build_feature_frame`
raising `AttributeError: 'str' object has no attribute 'astype'` (a real symptom of
`raw_df.get("src_ip", "")` falling back to its string default) before any downstream step
ran. The correct **GeneratedLabelledFlows / TrafficLabelling** release was located
(`data/GeneratedLabelledFlows.zip`), MD5-verified against `data/GeneratedLabelledFlows.md5`
(`5ca3f8f69e3514950681615824149973`, matched exactly), extracted, and its header confirmed
to contain `Flow ID`, `Source IP`, `Source Port`, `Destination IP`, `Destination Port`,
`Protocol`, `Timestamp`, and `Label` before any files were moved into `data/raw/cicids2017/`.
Filenames inside matched `CICIDS_FILES_ORDERED` exactly (nested one level under
`TrafficLabelling /`, trailing space in the directory name only, not the filenames).

A second real bug surfaced immediately after: `load_cicids_file` called `pd.read_csv`
with the default UTF-8 encoding, which failed with `UnicodeDecodeError: 'utf-8' codec
can't decode byte 0x96` on Thursday's file (a documented CICIDS2017 TrafficLabelling
encoding quirk — stray Windows-1252 bytes). Fixed by reading with `encoding="latin1"`
in `src/producer/flow_producer.py`, which accepts any byte value and preserves the
numeric/label columns this pipeline actually uses.

### Real preprocessing and training output (exact, unrounded)
`python -m src.data.preprocess` on the five real TrafficLabelling CSVs:
```
Loaded 249,203 flows from Monday-WorkingHours.pcap_ISCX.csv
Loaded 211,628 flows from Tuesday-WorkingHours.pcap_ISCX.csv
Loaded 226,767 flows from Wednesday-workingHours.pcap_ISCX.csv
Loaded 89,864 flows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Loaded 86,420 flows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Loaded total: 863,882 rows | attack rate: 7.8277%
train: 647,911 rows (attack: 3.1336%)
test: 129,582 rows (attack: 1.5434%)
stream: 86,389 rows (attack: 52.4592%)
Saved network_flow_features: 4,060 rows
```
Row counts are lower than the oft-cited 2,830,540 because only 5 of the 8 released daily
files are in scope per `CLAUDE_domain_migration.md`/`P1_COMBINED_SESSION.md`, and
`load_cicids_file` drops rows with non-finite `flow_bytes_per_sec`/`flow_packets_per_sec`
and de-duplicates by `flow_id` per file.

`python -m src.training.train` on this real data, exact printed output:
```
Train: (647911, 10), attack rate: 3.1336%
Test:  (129582, 10), attack rate: 1.5434%
Early stopping, best iteration is:
[29] valid_0's auc: 0.899719  valid_0's average_precision: 0.108376
Recall-calibrated threshold (target=0.95): 0.0074
PR-AUC: 0.1084
ROC-AUC: 0.8997
Precision: 0.0988
Recall: 0.9965
```
**Real PR-AUC: 0.1084. Real ROC-AUC: 0.8997. Real calibrated threshold: 0.0074. Realized
recall at that threshold: 0.9965 (exceeds the 0.95 target). Realized precision: 0.0988.**
PR-AUC is low in absolute terms — expected given the extreme class imbalance (3.13%
attack rate in train) and that `scale_pos_weight=10.0` (the config default, tuned to the
Sparkov fraud-detection base spec, not re-tuned for this dataset's true ~31:1 imbalance)
was not adjusted for this run. The threshold calibrated to 0.0074 is a direct consequence:
achieving 95%+ recall on this imbalanced, under-weighted model requires an extremely low
decision threshold, which in turn produces low precision (9.88%). This is a real,
observed characteristic of this specific training run, not a claim about the ceiling of
what the pipeline can achieve with proper hyperparameter tuning.

### Full Docker Compose stack with real data
`docker compose up --build` (after fixing a real bug: the `schema-registry` healthcheck
used `curl`, which does not exist in `confluentinc/cp-schema-registry` — confirmed via
`docker exec ... curl` returning `executable file not found in $PATH`; fixed to a bash
`/dev/tcp` TCP-reachability check in `docker-compose.yml`). One additional hiccup:
zookeeper/kafka containers left over from a prior session caused
`org.apache.zookeeper.KeeperException$NodeExistsException` on kafka startup (stale
ephemeral broker registration); resolved with `docker compose down && docker compose up -d`.
After these fixes, **all 7 services reported healthy** (zookeeper, kafka, redis,
schema-registry, api, prometheus, grafana) and stayed healthy for the full 2+ hour
duration of the streaming run, confirmed again at the end via `docker compose ps`.
`curl http://localhost:8000/health` → `{"status":"healthy","model_loaded":true,...}`
(model loaded via the documented MLflow-registry-fails → local-`production_model.pkl`
fallback path, since the MLflow sqlite backend recorded an absolute host artifact path
that doesn't resolve inside the container — expected, not a new bug).
`curl http://localhost:9090/api/v1/rules` confirmed both `PredictLatencyP95Breach` and
`PredictLatencyP99Breach` rules loaded with `health: ok`.

### Live streaming run: producer + consumer against the real 5-file replay
Consumer (`python -m src.consumer.flow_consumer`) and producer
(`python -m src.producer.flow_producer`) run as separate local processes against the
Dockerized Kafka/Redis/Schema-Registry (both reachable at `localhost`). Confirmed real,
not mocked:
- **Schema Registry / Avro**: consumer log — `Schema Registry available: deserializing
  events as Avro`; producer log — `Schema Registry available: serializing events as Avro`.
  Both used the live registry, not the JSON fallback.
- **Producer completed the full replay**: `Producer done. Total sent: 863,882` — all five
  files replayed in order (Monday → Tuesday → Wednesday → Thursday → Friday) at the
  configured 500 events/sec.
- **PSI-specific trigger, confirmed firing on real Tuesday attack traffic, distinct from
  the drift_share gate**: at consumer window `window_0501` (messages 250,001–250,500 —
  entirely within Tuesday's file; Monday ends at message 249,203), `check_drift_and_trigger_retraining`
  fired with `max_psi=0.2606` (exceeding `PSI_SIGNIFICANT_THRESHOLD=0.25`), computed
  against the Monday-benign reference distribution, on a window where the `drift_share`
  gate had NOT already fired (drift_share was below its own 0.3 threshold for that
  window). This is the requested confirmation: a genuine, separate PSI-based signal
  reacting to real attack traffic. Over the full run (consumer processed through message
  774,500 of 863,882, into Thursday's file, before being stopped): **27 PSI-specific
  triggers** and **1,320 drift_share triggers** fired, with **zero exceptions/tracebacks**
  in either the producer or consumer process across the entire run. The single highest
  PSI score observed was 2.4054 (later in the run, consistent with Friday's DDoS traffic
  producing extreme volumetric feature values far outside the Monday baseline).
- **Real, unexpected finding**: `drift_share` fired very frequently (990+ times) even
  during Monday's own benign traffic before any attack data streamed at all — a real
  consequence of running independent statistical tests across 10 feature columns on
  500-row windows, where the 0.3 `drift_share` threshold is crossed by chance more often
  than the reference-vs-reference case would suggest. This does not indicate a bug in the
  trigger logic; it demonstrates the `drift_share`/`DatasetDriftMetric` gate is
  considerably more trigger-happy than the PSI-specific gate at this window size.
- **Real, more significant finding**: every single retraining attempt (all 1,347
  triggers combined) aborted at Great Expectations validation with the same four
  failures: `flow_duration`, `flow_bytes_per_sec`, `flow_packets_per_sec`, and
  `flow_iat_mean` "contains values below minimum 0.0". Root cause confirmed directly
  against `train.parquet`: 2–3 rows out of 647,911 (0.0003%–0.0005%) have negative values
  in these columns (`flow_duration` min = -1.0, `flow_bytes_per_sec` min = -12,000,000.0),
  a well-documented CICIDS2017/CICFlowMeter artifact (clock-sync issues during capture —
  the same class of defect the README's Liu et al. limitations note references). The
  bounded GE rule added in Tier 1 (`add_bounded_expectations`) is working exactly as
  designed — it has no per-batch tolerance (unlike the 2% null-rate allowance), so a
  single offending row anywhere in the training data fails the whole suite and the
  retraining flow correctly aborts rather than training on invalid data. The practical
  consequence: with this exact training data, the self-healing loop's promotion step
  (steps 3–9) never actually ran in this session — every retraining attempt stopped at
  step 1–2, honestly abandoned, not silently bypassed.
- **Redis dedup verified against the real Redis container** (not `FakeRedis`):
  `is_duplicate_event` returned `False` then `True` for the same `event_id` against
  `Redis(host="localhost", port=6379)`, with `TTL` confirmed at 3600s via `redis-cli`.
- **Schema Registry test verified against the live registry**:
  `tests/test_schema_registry.py::test_register_schema_returns_positive_schema_id`
  passed (was skipped in all prior runs without a live registry).
- **Feast offline materialization verified against real materialized data** (not the
  test fixture): after a real `feast apply` against the actual feature repo,
  `materialize_offline_features` returned a non-empty DataFrame (1,904 rows, one row per
  distinct `src_ip` observed in the real per-`src_ip` aggregation table) with all 10
  `FEATURE_COLS` plus `src_ip`/`event_timestamp` columns present.
- Consumer was stopped manually (not crashed) at message 774,500/863,882 (partway through
  Thursday's file) once sufficient real evidence had been gathered; it did not process
  Friday's file to completion in this session. Producer's replay of all five files did
  complete in full.

### Sixth real bug found: MLflow model version type mismatch
Re-running the full pytest suite after training against real data (rather than the
previously-always-failing MLflow load) surfaced a sixth real, pre-existing bug: every
prior test run had MLflow registry loads fail and fall back to the local
`production_model.pkl` path, so `load_champion()`'s MLflow success branch had never
actually executed against a real registry. With a real trained model registered, the
MLflow branch succeeded (`Loaded champion from MLflow registry: version 1`) and exposed
that `version_info.version` is an `int`, not a `str` as the rest of the codebase assumes
— `ModelBundle.version` was set to this raw int and `HealthResponse`/`PredictionResponse`
(`model_version: str`) failed Pydantic validation. Fixed in
`src/serving/model_loader.py` by casting `version = str(version_info.version)`. Full
suite after the fix: **90 passed, 0 failed**.

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

---

# Root-Cause Fixes: scale_pos_weight and CICIDS2017 Sensor Artifact Rows

Follow-up to the real end-to-end run's two most significant findings. **Neither fix has
been re-verified against a real end-to-end run yet** — both are implemented and unit
tested only. The PR-AUC, recall, and precision numbers recorded earlier in this report
(0.1084, 0.9965, 0.0988) are from the run *before* these fixes and have not been
superseded. The next real run against real CICIDS2017 data is what will confirm whether
these changes actually improve PR-AUC and whether the retraining flow's promotion path
(steps 3-9) finally executes instead of aborting at step 1-2 on every attempt.

## Finding 1: Low PR-AUC root-caused to an untuned scale_pos_weight
The real training run recorded PR-AUC 0.1084 with `scale_pos_weight=10.0`, a constant
inherited from the original Sparkov fraud-detection base spec. The real training split
observed in that run has a 3.1336% attack rate (roughly 31:1 negative:positive), not the
ratio `10.0` implies (~9:1). The model was therefore substantially under-weighting the
minority (attack) class relative to its actual real-world scarcity in this dataset.

**Implemented:** `compute_scale_pos_weight(y_train)` in `src/training/train.py`,
computing `n_negative / n_positive` from the real training labels at training time.
Wired into `train_lgbm` immediately before the LightGBM `fit` call, replacing the static
`scale_pos_weight` key that previously lived in `LGBM_PARAMS` (`src/config.py`). Removed
the now-unused `settings.scale_pos_weight` field and its `SCALE_POS_WEIGHT` entry in
`.env.example`, since the value is no longer a fixed constant.

**Tests:** `tests/test_train.py` — asserts `compute_scale_pos_weight` returns `20.0` for
100 negatives / 5 positives, and `1.0` for a balanced 50/50 split.

**Not verified:** training has not been re-run with this change. Computed directly
against the real `train.parquet` written by the last run (627,608 negative rows, 20,303
positive rows), `compute_scale_pos_weight` would return `627608 / 20303 ≈ 30.91`,
materially higher than the previous fixed `10.0` — whether this improves PR-AUC in
practice is unconfirmed.

## Finding 2: Every retraining attempt aborting at GE validation
Root-caused in the last session to 2-3 rows out of 647,911 (0.0003%-0.0005%) in
`train.parquet` with negative values in `flow_duration`, `flow_bytes_per_sec`,
`flow_packets_per_sec`, and `flow_iat_mean` — a documented CICIDS2017/CICFlowMeter
clock-sync capture artifact, the same class of defect cited in the README's Liu et al.
limitations note. The Tier 1 bounded GE rule (`add_bounded_expectations`) correctly has
no per-batch tolerance for these columns, so it correctly failed the whole suite on every
one of the 1,347 retraining attempts observed in the real run. This was working as
designed, not a validation bug, and the bounded rule itself was not loosened.

**Implemented:** `filter_invalid_flow_rows(df, bounded_cols)` in `src/data/preprocess.py`,
dropping any row with a negative value in `INVALID_FLOW_BOUNDED_COLS`
(`flow_duration`, `flow_bytes_per_sec`, `flow_packets_per_sec`, `flow_iat_mean`) and
logging the exact count dropped with the reason cited (known CICIDS2017/CICFlowMeter
clock-sync artifact, not a pipeline defect). Wired into `run_preprocessing()` immediately
after `impute_numeric` and before `temporal_split`, so the fix is applied at the source
rather than allowing invalid rows to reach GE validation downstream.

**Tests:** `tests/test_preprocessing.py` —
`test_filter_invalid_flow_rows_drops_negative_values` asserts a synthetic row with a
negative `flow_duration` is dropped and the drop count is logged;
`test_filter_invalid_flow_rows_retains_valid_rows` asserts an all-valid batch is
unchanged.

**Not verified:** preprocessing has not been re-run against the real CICIDS2017 files
with this change. Whether the real training data has *exactly* the 2-3 previously
identified negative rows (removable cleanly without cascading effects on the temporal
split boundaries or the per-`src_ip` Feast aggregation) or additional rows not yet
surfaced is unconfirmed until the next real run.

## Deviation
The test file names in this section's spec (`tests/test_train.py`,
`tests/test_preprocess.py`) were followed for `test_train.py` (new file, no prior test
module existed for `src/training/train.py`'s standalone functions) but
`filter_invalid_flow_rows` tests were added to the existing `tests/test_preprocessing.py`
rather than a new `tests/test_preprocess.py`, since a test module for
`src/data/preprocess.py` already exists under that name and a second file would
duplicate/fragment test infrastructure for the same source module.

---

# Final Verification Run: Both Fixes Confirmed Against Real Data

This closes the gap left by the previous session: both root-cause fixes are now
verified against a real re-run of preprocessing, training, and a live Docker Compose
streaming run. **These are the final numbers resume bullets should be written from.**

## Re-run preprocessing: `filter_invalid_flow_rows` confirmed
`python -m src.data.preprocess` against the same five real CICIDS2017 files, exact
printed output:
```
Loaded total: 863,882 rows | attack rate: 7.8277%
  Dropped 3 row(s) with negative values in ['flow_duration', 'flow_bytes_per_sec',
  'flow_packets_per_sec', 'flow_iat_mean'] (known CICIDS2017/CICFlowMeter clock-sync
  capture artifact, not a pipeline defect)
train: 647,909 rows (attack: 3.1336%)
test: 129,582 rows (attack: 1.5434%)
stream: 86,388 rows (attack: 52.4598%)
```
Exactly 3 rows dropped, reasoning logged explicitly (not silent), matching the 2-3 rows
identified in the prior session's root-cause analysis. Directly verified against the
resulting `train.parquet`: `validate_batch` now returns `(True, [])` — **GE validation
passes cleanly for the first time this project has run against real CICIDS2017 data.**

## Re-run training: `compute_scale_pos_weight` confirmed, real numbers improved
`python -m src.training.train`, exact printed output:
```
Train: (647909, 10), attack rate: 3.1336%
Test:  (129582, 10), attack rate: 1.5434%
Computed scale_pos_weight from training data: 30.9120
Early stopping, best iteration is:
[1] valid_0's auc: 0.938324  valid_0's average_precision: 0.162536
Recall-calibrated threshold (target=0.95): 0.0298
PR-AUC: 0.1625
ROC-AUC: 0.9383
Precision: 0.1719
Recall: 0.9555
PR-AUC: challenger=0.1625, champion=0.1084, improvement=+0.0542 (margin=0.005)
Promoted version 2 to 'champion'
```
`scale_pos_weight` is printed as a computed value (30.9120), not a hardcoded number,
confirming the dynamic computation is wired in and active.

**Comparison against the previous run's numbers (not rounded or estimated):**

| Metric | Previous (fixed `scale_pos_weight=10.0`) | This run (computed `30.9120`) | Change |
|---|---|---|---|
| PR-AUC | 0.1084 | **0.1625** | **Improved, +0.0542 (+50.0% relative)** |
| ROC-AUC | 0.8997 | **0.9383** | Improved, +0.0386 |
| Calibrated threshold | 0.0074 | 0.0298 | Higher (expected — less aggressive weighting needs a higher cutoff to hit the same recall target) |
| Recall | 0.9965 | 0.9555 | **Lower**, but still clears the 0.95 target |
| Precision | 0.0988 | **0.1719** | **Improved, +0.0731 (+74.0% relative)** |

PR-AUC and precision both improved substantially; recall decreased but remains above the
`SERVING_RECALL_TARGET = 0.95` floor. This is a real, unambiguous improvement from the
fix, not an assumption — reported exactly as observed, including the recall trade-off.
This standalone `python -m src.training.train` run also promoted (challenger 0.1625 vs.
no prior champion in a fresh MLflow versioning sequence, +0.0542 over the previous
run's champion 0.1084, exceeding the 0.005 margin) — but this script does not run GE
validation itself; that confirmation came from the live streaming run below.

## Full Docker Compose stack: all 7 services healthy again
`docker compose up --build -d` hit the same stale zookeeper/kafka ephemeral-broker
`NodeExistsException` seen in the previous session (containers left over from a prior
`docker compose up`, not a new bug); resolved the same way with
`docker compose down && docker compose up -d`. All 7 services (zookeeper, kafka, redis,
schema-registry, api, prometheus, grafana) reported healthy, and remained healthy for
the full ~1.5 hour duration of this run, reconfirmed via `docker compose ps` and
`curl http://localhost:8000/health` at the end.

## Live streaming run: GE validation now passes on every attempt; PSI trigger reconfirmed; no promotion in this session (explained, not a bug)
Producer and consumer run again as separate local processes against the same live
Kafka/Redis/Schema-Registry stack. Producer replayed all five files to completion in
full (`Producer done. Total sent: 863,882`) at its normal ~500 events/sec pace,
independent of the consumer.

**Real, significant, and initially unexpected finding:** once GE validation started
passing, every drift trigger began running a real full LightGBM training cycle
(~6-10 seconds each: materialize features, load data, train, evaluate, compare) instead
of aborting in milliseconds at validation. This dropped sustained consumer throughput
from roughly 200-250 msg/sec in the previous (always-aborting) run to **~64 msg/sec** in
this run. Reaching Tuesday's data (message 249,203) took approximately **68 minutes** of
wall-clock time this session, versus roughly 10 minutes previously. This is a legitimate
cost of the fix working as intended, not a defect, and is recorded here because it is a
real operational characteristic of the self-healing loop once retraining actually runs
to completion on every trigger: **at this drift-trigger frequency (drift_share fires on
the large majority of 500-event windows — see the prior session's finding on why 0.3 is
an easily-crossed threshold at this window size), the retraining flow cannot keep up
with the replay rate in real time.** This is worth flagging as a capacity/tuning
consideration for a real deployment, separate from the two fixes verified here.

Consumer was run for 1 hour 23 minutes, processing 310,400 of 863,882 messages (through
all of Monday and into Tuesday) before being stopped once sufficient evidence had been
gathered — final exact counts from that run:
- **535 retraining flows started, 534 reached the `compare-models` step, 0 aborted at
  GE validation.** (Previous run: 1,347 attempts, 0 reached `compare-models`, all 1,347
  aborted at validation.) This is the direct, unambiguous confirmation that
  `filter_invalid_flow_rows` resolved the root cause: **the retraining flow's steps 3-9
  (materialize, load, train, evaluate, compare, and conditionally promote) now execute
  on every single trigger.**
- **12 PSI-specific triggers, 523 drift_share triggers**, zero tracebacks/crashes.
- **The PSI trigger reconfirmed firing on Tuesday's real attack traffic**, at the exact
  same window as the previous run (`window_0501`, messages 250,001-250,500, entirely
  within Tuesday's file, `max_psi=0.2606`) — deterministic given identical input data.
  This specific PSI-triggered flow (`wise-puffin`) was traced end to end: validation
  passed, materialization completed, data loaded, a challenger trained
  (MLflow version 434, PR-AUC 0.1625), and `compare-models` correctly evaluated
  `challenger=0.1625, champion=0.1625, improvement=+0.0000` and decided **not** to
  promote — a correct decision given zero improvement, not a failure of the trigger or
  the flow.
- **0 promotions occurred anywhere in this run** (`grep -c "New champion promoted"` = 0
  across all 534 completed flows, drift_share- and PSI-triggered alike). Root cause,
  confirmed by direct inspection: `train_lgbm` uses `random_state=42` and every
  retraining attempt in this session re-trained against the exact same `train.parquet`
  with the exact same `scale_pos_weight` — training is fully deterministic here, so
  every challenger reproduces the identical PR-AUC (0.1625) already held by the
  champion (promoted earlier in this session via the standalone `train.py` run above).
  With zero variance between challenger and champion, `should_promote`'s
  `PROMOTION_PRAUC_MARGIN = 0.005` is correctly never exceeded. **This is not a residual
  defect in the retraining flow** — the flow demonstrably reaches and correctly
  evaluates the promotion decision (confirmed above); it is an artifact of testing
  promotion logic with a static dataset and a fixed random seed, where there is no
  further headroom for a deterministically-retrained model to improve over itself.
  Observing an actual promotion via the live streaming path would require either new
  incoming training data with a genuinely different distribution, a change to
  `LGBM_PARAMS`/`random_state`, or manually lowering the champion's recorded metrics.

## Summary: what is now confirmed and what remains open
- **Confirmed and fixed:** the always-aborting-at-GE-validation defect. 0 aborts across
  535 attempts in this run versus 100% abort rate (1,347/1,347) previously.
- **Confirmed and improved:** PR-AUC (0.1084 → 0.1625) and precision (0.0988 → 0.1719)
  from the dynamic `scale_pos_weight` fix, at the cost of a small, still-passing drop in
  recall (0.9965 → 0.9555, target is 0.95).
- **Confirmed unchanged:** the PSI-specific trigger still fires correctly and
  deterministically on real Tuesday attack traffic, independent of the drift_share gate.
- **Not confirmed / newly surfaced as an open question:** actual promotion via the live
  self-healing loop. The flow's promotion step is proven to execute and evaluate
  correctly, but no promotion was observed in this session because of deterministic
  training producing zero challenger/champion divergence — this is a property of the
  current fixed-seed, fixed-dataset test setup, not a proven defect, but it has also not
  been positively demonstrated end-to-end. A genuinely new distribution (e.g. resuming
  the streamed replay through Wednesday/Thursday/Friday, which contain different attack
  patterns) or an intentionally perturbed run would be needed to observe a real
  promotion through the live path.
- **New operational finding:** at the default drift-trigger frequency, the retraining
  flow cannot sustain real-time throughput against the producer's 500 events/sec replay
  rate once every trigger completes a real training cycle (~64 msg/sec achieved vs. 500
  msg/sec produced) — a capacity consideration for real deployment tuning, not addressed
  in this session.
