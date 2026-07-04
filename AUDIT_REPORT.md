# P1 Combined Session — Phase 0 Repo Audit

Audit of current repo state against `CLAUDE_domain_migration.md` (13 sections) and the
8 infra improvements defined in `P1_COMBINED_SESSION.md` (Phases 2-4). Read-only
investigation, no code changes made.

`git log --oneline -20` and `git status` were reviewed: work is on `main`, staged
deletions of stale fraud artifacts (models/, old kafka_producer.py/kafka_consumer.py,
drift_injector.py, fraud_checkpoint.json), and a large set of unstaged modifications plus
untracked new files (`src/schemas/`, `src/producer/flow_producer.py`,
`src/consumer/flow_consumer.py`, `scripts/`, `IMPLEMENTATION_REPORT.md`, test files) —
consistent with an in-progress domain migration that has not yet been committed.

---

## Domain Migration Sections (`CLAUDE_domain_migration.md`)

| # | Section | Verdict | Note |
|---|---------|---------|------|
| 1 | Event Schema | **DONE** | `src/schemas/network_flow_event.avsc` has all 21 fields matching spec exactly. Old `transaction_event.avsc` absent. |
| 2 | Kafka Topic Names | **DONE** | `src/config.py:17-18` — `kafka_topic="network_flows"`, `kafka_dlq_topic="network_flows.dlq"`. No lingering `transactions` references. |
| 3 | Kafka Producer | **DONE** | `src/producer/flow_producer.py` exists; old `transaction_producer.py`/`kafka_producer.py` deleted. |
| 4 | Feature Engineering | **DONE** | `src/features/engineering.py` defines `FEATURE_COLS` (10 features); `config.py` imports it, `CATEGORICAL_COLS=[]`. `fwd_bwd_packet_ratio` epsilon-guard covered by `test_features.py`. |
| 5 | Feast Entity/Feature View | **DONE** | `entities.py`: `network_entity` on `src_ip`. `feature_views.py`: `network_flow_features`, TTL `timedelta(hours=1)`, 10 features. |
| 6 | GE Column Assertions | **DONE** | `src/validation/expectations.py`: `SUITE_NAME="network_flow_feature_suite"`, `NUMERICAL_FEATURE_COLS=list(FEATURE_COLS)`, `MAX_NULL_RATE=0.02`. |
| 7 | Evidently Column References | **DONE** | `drift_detector.py` imports `FEATURE_COLS` from `engineering.py`; `REFERENCE_DATA_PATH` points at `reference_network_flows.parquet`. |
| 8 | MLflow Experiment Name | **DONE** | `config.py:27` — `mlflow_experiment_name="network_anomaly_detection"`, used in `train.py:188`. |
| 9 | Operating Point (PR-AUC replacing F2) | **DONE** | `train.py`/`threshold.py` use PR-AUC and `calibrate_threshold` (recall-targeted); no F2 logic remains. |
| 10 | Consumer Update | **DONE** | `src/consumer/flow_consumer.py` uses `network_flows` topic, `src_ip`-keyed events, calls `compute_features()`. |
| 11 | Files to Rename | **DONE** | No `transaction_producer.py`/`transaction_consumer.py`/`transaction_event.avsc` remain anywhere. |
| 12 | README Update | **PARTIAL** | Dataset/feature/operating-point sections updated correctly (no "fraud detection" or "IoT sensor telemetry" strings remain). **Missing**: the required honest limitations note citing Liu et al. on CICIDS2017 labeling/flow-construction issues — zero matches for "limitation" or "Liu et al" in README.md. |
| 13 | Reference Dataset Generation | **DONE** | `scripts/generate_reference_dataset.py` exists with correct constants and imports. |

**Summary:** 12 of 13 sections DONE. Section 12 (README) is PARTIAL — missing only the
limitations note. `IMPLEMENTATION_REPORT.md` already exists and its claims check out
accurately against the real files; it does not address the 8 infra improvements (it
predates this combined session and is correctly scoped to migration only).

---

## Infra Improvements (`P1_COMBINED_SESSION.md`, Phases 2-4)

| # | Improvement | Verdict | Note |
|---|-------------|---------|------|
| 1 | DLQ Hardening | **PARTIAL** | `src/consumer/dlq_handler.py` has `send_to_dlq` (not `route_to_dlq`) publishing to `settings.kafka_dlq_topic` with a header, but the header key is `"error"`, not the spec's `DLQ_ERROR_HEADER_KEY="validation_error"`. No `DLQ_ERROR_HEADER_KEY` constant exists. |
| 2 | PSI Drift Detection → retraining trigger | **PARTIAL** | `alert_handler.py` already triggers `retraining_flow()` on drift (not just logging), but gating is via Evidently's `drift_share`/`DatasetDriftMetric`, not PSI specifically. No `PSI_MINOR_THRESHOLD`/`PSI_SIGNIFICANT_THRESHOLD` constants or `check_drift_and_trigger_retraining` function exist. |
| 3 | Latency SLOs | **MISSING** | No `monitoring/prometheus/alerts.yml`. `monitoring/prometheus.yml` has no `rule_files:` section. No SLO/burn-rate panel in `infra.json`. |
| 4 | Bounded GE Rules | **MISSING** | `expectations.py` has only null-rate and row-count checks. No `FEATURE_BOUNDS` dict or `add_bounded_expectations` function anywhere. |
| 5 | Idempotent Consumer / Redis Dedup | **MISSING** | `flow_consumer.py` has zero Redis usage. No `is_duplicate_event`, `DEDUP_KEY_PREFIX`, or SETNX logic. Redis only appears in `feature_store.yaml` as the Feast online store, unrelated. |
| 6 | Shadow Mode Champion-Challenger | **MISSING** | No `score_shadow_mode`/`evaluate_shadow_divergence` functions anywhere. Note: `healing_mode="SHADOW"` is an existing, unrelated concept (means "trigger retraining, register challenger only" — not "score live traffic with both models and log divergence"). Do not treat this as partial credit. |
| 7 | Confluent Schema Registry | **MISSING** | `docker-compose.yml` (confirmed present) has no `schema-registry` service. No `register_schema` function, no `AvroSerializer`/`AvroDeserializer` usage. Producer/consumer currently do plain JSON serialization despite the static `.avsc` file existing. |
| 8 | Feast Offline Materialization | **PARTIAL** | `src/features/materializer.py` exists (base spec's online/offline materialization script) but `src/orchestration/flows/materialization_flow.py` does not exist — only `retraining_flow.py` is present under `src/orchestration/flows/`. No `materialize_offline_features` function or `MATERIALIZATION_LOOKBACK_HOURS` constant. |

**Summary:** 0 of 8 infra improvements are fully DONE. 3 are PARTIAL (1, 2, 8), 5 are
MISSING (3, 4, 5, 6, 7).

---

## Testing Requirements Coverage

No dedicated tests exist yet for any of the 8 infra improvements (consistent with none
being fully DONE). Existing incidental matches — `test_config.py` asserting
`kafka_dlq_topic`/`healing_mode` enum values, `test_features.py`'s epsilon-guard test —
are not the spec-mandated tests (DLQ header assertion, PSI-trigger assertion, alert-rule
YAML validity, GE bounds rejection, dedup TTL behavior, shadow-mode disagreement
handling, schema registry ID assertion, materialization completion assertion).

---

## Other Observations

- `docker-compose.yml` exists at repo root (2.7KB) and is unmodified/tracked — it
  correctly did not appear in `git status` since there are no working-tree changes to it.
- `network_flow_checkpoint.json` GE checkpoint exists with the correct name.
- Section 12 (README limitations note) and infra improvements 3-7 are the only gaps
  remaining before this session's checklist is complete.

---

## Next Steps (per user instruction: awaiting confirmation before proceeding)

- **Phase 1**: Finish README section 12 (limitations note only — everything else in the
  migration is DONE).
- **Phase 2 (Tier 1)**: Fix DLQ header key/constant (#1), formalize PSI-specific
  trigger function/constants (#2), add latency SLO alert rule file + Grafana panel (#3),
  add bounded GE expectations (#4).
- **Phase 3 (Tier 2)**: Add Redis-backed idempotent consumer dedup (#5), add shadow-mode
  scoring/divergence evaluation (#6).
- **Phase 4 (Tier 3)**: Add Confluent Schema Registry wiring (#7), add
  `materialization_flow.py` with `materialize_offline_features` (#8).

Stopping here per instructions — no code will be written until this audit is confirmed.
