# Design decisions

Each entry follows the template in the project conventions. Dates are the day the decision was
taken during the v2 build.

## D-001: Package layout `src/driftguard/` and removal of the flat layout
Status: accepted (2026-10-14)
Context: the first version used a flat `src/<module>` layout whose consumer embedded training, and its tests targeted Prefect 2 and Great Expectations 0.18 APIs that do not exist in the pinned environment.
Decision: rebuild under `src/driftguard/` with the layout of the specification and port only the useful logic (threshold calibration, class weighting, ONNX export).
Alternatives: restyle the old modules in place (rejected: the consumer's training path is the defect being removed and old tests would keep CI red).
Consequences: history before the rebuild is only in git; no test from the first version survives.

## D-002: uv with exact pins and a lockfile
Status: accepted (2026-10-14)
Context: the first version pinned most packages but had no lockfile and a broken local environment.
Decision: `pyproject.toml` pins every runtime and dev dependency to an exact version and `uv.lock` records the resolution for Python 3.11.
Alternatives: pip with `requirements.txt` (rejected: no lock of transitive dependencies), Poetry (rejected: heavier for this use).
Consequences: upgrades happen in dedicated commits; Feast caps `redis` below 5.

## D-003: Idempotency with claim leases
Status: accepted (2026-10-14)
Context: a plain `SET NX` before scoring loses events if the consumer dies after claiming them but before producing predictions, which the kill-and-restart test would expose.
Decision: claim `dg:seen:{event_id}` as `pending:<owner>` with a 30 s lease, mark it `1` with a 48 h TTL after predictions are flushed and before the offset commit, and reclaim a pending key whose lease expired.
Alternatives: `SET NX` with the final TTL up front (rejected: loses events on a crash), Kafka transactions (rejected: ties scoring, Redis, and an HTTP call into one transaction that cannot be made atomic).
Consequences: after a crash, redelivered events wait up to one lease before they are reclaimed; a done key is a durable duplicate marker for 48 h.

## D-004: Shadow scoring from the consumer
Status: accepted (2026-10-14)
Context: shadow results must be on the prediction record and must not add latency to the champion path.
Decision: in shadow mode the consumer sends each batch to `/score` and, concurrently, to `/score/shadow`; the champion latency is the response latency, and both scores go into one `predictions.v1` record. A shadow failure leaves `shadow_score` null and increments a metric.
Alternatives: the serving router calls the challenger itself and returns both (rejected: the response then includes challenger work unless it is made asynchronous with a side channel to Kafka).
Consequences: the consumer carries a small thread pool for shadow calls.

## D-005: Rollout controller in the orchestrator, aliases as the control plane
Status: accepted (2026-10-14)
Context: promotion and rollback must be reversible with one command and must not redeploy anything.
Decision: serving loads models by MLflow alias (`champion`, `challenger`, `previous_champion`) and swaps them atomically from a 10 s watcher; the orchestrator runs the shadow gate and canary guardrails and moves aliases; every move is appended to `results/promotions.jsonl`.
Alternatives: model files baked into images (rejected: promotion would be a rollout), a separate rollout service (rejected: a thirteenth service with no extra guarantee).
Consequences: shadow pairs live in the controller's memory, so a restart during shadow restarts the collection.

## D-006: Partitioned parquet behind the Feast PushSource
Status: accepted (2026-10-14)
Context: the specification asks the first build hour to test whether Feast's file offline store accepts `ONLINE_AND_OFFLINE` pushes. Measured with Feast 0.49: it works only for a single-file batch source and rewrites that whole file on every push; a partitioned directory source fails.
Decision: push features online through a `PushSource`, and append the same rows to date-partitioned parquet that is the PushSource batch source, which keeps the training-serving guarantee (training reads the served values with point-in-time joins).
Alternatives: push to a single parquet file (rejected: O(file) rewrite per batch), a database offline store (rejected: another service).
Consequences: offline rows are written by the consumer rather than by Feast; a test records the Feast behavior.

## D-007: Asynchronous retraining through a Prefect deployment
Status: accepted (2026-10-14)
Context: synchronous retraining inside the consumer reduced throughput from about 250 to about 64 msg/s.
Decision: the monitor publishes drift events; the orchestrator submits the `retrain/retrain-on-drift` deployment with `timeout=0` and a concurrency limit of 1 with collision strategy `CANCEL_NEW`.
Alternatives: a thread pool inside the consumer (rejected: shares CPU with inference), Celery (rejected: a second orchestration system).
Consequences: retraining no longer blocks ingestion; promotion becomes eventually consistent.

## D-008: Threshold selection
Status: accepted (2026-10-14)
Context: the specification asks for "the lowest threshold that reaches `target_recall`". Read literally, the lowest threshold of all reaches recall 1.0 and is useless.
Decision: scan thresholds from high to low and take the first at which recall reaches the target, which is the strictest threshold that meets it and so maximizes precision under the recall floor.
Alternatives: the lowest threshold overall (rejected: trivial), a fixed 0.5 (rejected: ignores class imbalance).
Consequences: the threshold is chosen on a validation slice only and stored with each model version.

## D-009: Great Expectations bounds fitted once
Status: accepted (2026-10-14)
Context: bounds that move to make a run pass defeat the check.
Decision: `scripts/fit_gx_bounds.py` fits per-feature bounds at the 0.1 and 99.9 percentiles of the initial reference data once, writes them to `configs/data/<dataset>_gx_bounds.json`, and refuses to overwrite without `--o-force`; at least 99% of rows must lie within bounds. Bounds are never loosened.
Alternatives: refit on every window (rejected: hides drift), hard min and max (rejected: one outlier fails everything).
Consequences: the bounds file is committed after the first real fit; CI uses synthetic bounds that are never committed.

## D-010: Monitor windows over two streams
Status: accepted (2026-10-14)
Context: flows and predictions arrive on different topics at different times.
Decision: count windows close when both streams have filled the same index range; Kyoto month windows close after 1,000 later events in both streams, and later arrivals for a closed month are counted as late.
Alternatives: join flows and predictions by event id (rejected: needs unbounded state for the join).
Consequences: a stalled stream stalls window closing.

## D-011: Per-batch stream feature computation
Status: accepted (2026-10-14)
Context: merging HyperLogLogs for every event caps throughput near 2,000 events/s.
Decision: counts use `ZCOUNT (t-w, t]` per event; unique counts, mean bytes, and SYN share are computed once per source key per poll batch over the six buckets ending at that key's newest bucket; all writes and reads run in two pipelines per batch.
Alternatives: per-event merges (rejected: throughput), approximate counts without HyperLogLog (rejected: memory grows with the window).
Consequences: unique counts depend on batch composition, so replay features match served values up to that composition; training reads the exact served values from the offline store.

## D-012: E1 behavioral-feature ablation by replay
Status: accepted (2026-10-14)
Context: the ablation needs behavioral features for every flow of the week.
Decision: replay the dataset through the same computer against Redis under a private key prefix, with the consumer's batch size, then delete that state.
Alternatives: a separate offline implementation (rejected: two implementations can disagree).
Consequences: E1 with the ablation needs Redis and the IP salt.

## D-013: E2 runs offline
Status: accepted (2026-10-14)
Context: 20 seeds times every shift and magnitude is thousands of windows.
Decision: drive the production monitor core directly with shuffled base segments, one injected shift per run, and delayed labels at coverage 1.0 and 0.3; delay is measured in windows from the onset window.
Alternatives: run through Kafka (rejected: hours per cell with no change to the detector code under test).
Consequences: E2 measures detectors and policy, not transport.

## D-014: Redis, not Flink or Spark, for window state
Status: accepted (2026-10-14)
Context: per-key window state is small.
Decision: keep window state in Redis with TTLs of twice the window.
Alternatives: Flink or Spark Structured Streaming (rejected: a JVM stream processor adds an operationally heavy component without changing any guarantee this system makes).
Consequences: window correctness depends on the consumer's batch ordering and on Redis TTLs.

## D-015: Initial champion without the gate
Status: accepted (2026-10-14)
Context: the gate compares a challenger with an incumbent, and a fresh stack has none.
Decision: `scripts/bootstrap.py` trains the initial champion on the initial training period (Monday and Tuesday for CICIDS2017, because Monday has no attacks; 2006 and 2007 for Kyoto), registers it as `champion`, and logs it as `bootstrap` in `promotions.jsonl`.
Alternatives: serve a dummy model until the first retraining (rejected: scores would be meaningless).
Consequences: this is the only model that serves without passing the gate; it is stated as a limitation.

## D-016: E1 threshold validation slice
Status: accepted (2026-10-14)
Context: the last stretch of the training days may hold too few attacks to choose a threshold, and the test days must never be used.
Decision: E1 selects the threshold on a random stratified 10% of the training days (configurable to the newest 10% by event time); retraining uses the newest slice by event time as the specification states.
Alternatives: a time-ordered slice for E1 (available through configuration).
Consequences: E1's threshold is not time-ordered within the training days; the test days stay untouched.

## D-017: Load-test switch for stream features
Status: accepted (2026-10-14)
Context: E3 must report the throughput cost of stream features.
Decision: `STREAM_FEATURES_MODE=off`, read by the consumer, replaces the behavioral features with zeros so the request shape and the model input are unchanged while the Redis window work disappears. It is documented as load-test only.
Alternatives: a second model without the features (rejected: changes the scoring cost being compared).
Consequences: results produced with the switch off are keyed `stream_features=off` in the E3 file.

## D-018: Kubernetes validation without a cluster
Status: accepted (2026-10-15)
Context: a kind cluster is heavy and must run alone.
Decision: CI validates the chart with `helm lint`, `helm template`, and kubeconform using CRD schemas for Strimzi, KEDA, and the Prometheus Operator, and a nightly job runs a reduced E3b on kind with synthetic data.
Alternatives: no manifest validation outside a cluster (rejected: errors would surface only at install time).
Consequences: kubeconform downloads CRD schemas at CI time.
