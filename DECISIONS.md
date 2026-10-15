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

## D-011: Per-event stream feature computation in event-time order
Status: accepted (2026-10-14), revised (2026-10-15) after review
Context: the first implementation computed unique counts, mean bytes, and SYN share once per source per poll batch, and pruned all window state by the newest event of the batch before any read. Review reproduced two defects: early events in a long batch got undercounts (even 0 for their own flow; 57% of 500-event batches in the CICIDS2017 stream span more than 15 s), and unique counts changed for 85 to 98% of rows when the poll size changed from 500 to 37.
Decision: process each batch in event-time order and, for each event, write its state and then read its features inside one pipeline; prune once per key after all reads. Counts use `ZCOUNT (t - w, t]`; unique counts, mean bytes, and SYN share use the up to six 10 s buckets ending at the event's own bucket.
Alternatives: batch-level reads (rejected: depends on Kafka poll boundaries and looks ahead, contradicting the determinism guarantee), a separate offline reimplementation (rejected: two implementations can disagree).
Consequences: about 18 Redis commands per event instead of a few per batch, so the throughput cost of stream features (E3 features on and off) is higher than the specification's note about per-batch merges anticipated; features are identical for any batch split of an in-order stream.

## D-012: E1 behavioral-feature ablation by replay
Status: accepted (2026-10-14)
Context: the ablation needs behavioral features for every flow of the week.
Decision: replay the dataset through the same computer against Redis under a private key prefix, with the consumer's batch size, then delete that state.
Alternatives: a separate offline implementation (rejected: two implementations can disagree).
Consequences: E1 with the ablation needs Redis and the IP salt; replayed values equal served values for an in-order stream because features do not depend on batch boundaries (D-011).

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

## D-016: E1 validation by held-out blocks
Status: accepted (2026-10-15), revised after the first E1 run
Context: the first E1 run used a random stratified 10% of the training days for threshold selection and produced validation PR-AUC of about 1.0 and thresholds near 0.99999. Random slices put a temporal neighbor of every fit row into validation, and a newest-slice validation (as retraining uses) is unusable here because the last 60,000 training rows hold 7 attacks.
Decision: hold out whole contiguous blocks of the stream (every 10th block of 5,000 flows) for threshold selection. Test days stay untouched.
Result: blocked validation is the methodologically safer split but did not change the outcome: the threshold stayed at 0.999998 and validation PR-AUC stayed at 1.0. Leakage between fit and validation was therefore not the cause of the extreme threshold; see D-019 for the cause.
Alternatives: random slice (rejected: leaks neighbors), newest slice (rejected: 11 positives).
Consequences: validation contains only attack families seen in training, so recall on unseen families at the stored threshold can be low; E1 reports it per family.

## D-019: Model capacity chosen so the threshold keeps headroom below 1.0
Status: accepted (2026-10-15)
Context: with 400 trees, seen attackers score within about 3e-6 of 1.0, so the threshold that reaches recall 0.95 sits in a saturated tail. A seen family (SSH-Patator) with median 1 minus score of 5e-6 fell just past it and had recall 0.000, while a 100-tree model recovered it. The stored threshold was brittle to score shifts far smaller than any real drift. This was found by inspecting E1 output, which included test-day per-family recall, so the investigation saw test data.
Decision: choose `n_estimators` by a criterion fixed before the final choice and evaluated on validation only: the largest of {50, 100, 150, 200, 300, 400} (learning rate 0.05) whose blocked-validation threshold has at least 1e-3 of headroom below 1.0. That is 100 (headroom 5.5e-3; 150 trees gives 4.9e-4). Training also records `threshold_headroom` and logs a warning below 1e-3.
Alternatives: keep 400 trees and report the brittle operating point (rejected: the per-family recall would reflect float saturation, not the model), calibrate scores after training (rejected: adds a stage with no spec support), choose the count from test-day recall (rejected: tuning on the evaluation).
Consequences: validation ranking is unchanged (PR-AUC about 1.0); the contamination caveat above is disclosed in the README limitations.

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

## D-020: KSWIN on a subsampled score stream (policy version 4)
Status: accepted (2026-10-15)
Context: the first E2 smoke run (2 seeds, 24 windows) showed `kswin_score` flagging 68 to 92 of every 100 stationary windows and the combined policy raising 15 to 26 false alarms per 100 windows, about one spurious retraining trigger every four to five windows. Cause: KSWIN runs a KS test per update, so a 5,000-event window holds thousands of overlapping tests at alpha 0.005; on stationary continuous scores it fired about 4 times per 5,000 events in 40 of 40 windows. A window-level count threshold cannot fix this: KSWIN resets after each detection, and its count under a persistent strong shift (about 6 per 5,000 events) was no larger than under the null (about 4).
Decision: keep the specified parameters (alpha 0.005, window 500, statistic 100) and feed KSWIN one score in `stride` events, with stride 100 (`kswin.stride`), so a window holds about 50 tests. The policy version is 4 because the semantics of `kswin_score` changed. Measured on beta-distributed scores: 4.8% of stationary windows flagged, a strong shift detected in the first window in 8 of 8 runs, a mild shift within two windows in 6 of 8.
Alternatives: a calibrated detection-count limit (rejected: no power, see above), feeding window means (rejected: needs hundreds of windows of history), dropping KSWIN (rejected: the specification requires it).
Consequences: KSWIN needs about 50,000 events before its first test; the strided stream spans more event time than the specification's wording implies. E2 reports the resulting rates.

## D-021: E2 applies the same GX gate as the deployed monitor
Status: accepted (2026-10-15)
Context: the deployed monitor blocks any window that fails the Great Expectations suite and pauses retraining, but E2 originally ran the policy with no suite, so its combined-policy delay and miss rate described a system that does not exist. A covariate shift large enough to push more than 1% of a feature outside its fitted 0.1 and 99.9 percentile bounds is classified as a data-quality failure, not drift.
Decision: E2 builds a GX validator per dataset from bounds fitted on the reference rows (the same fit as `scripts/fit_gx_bounds.py`, `mostly` and the minimum rows from the dataset config) and runs the policy with it. Each shift also reports the clean-window failure rate (failures per 100 windows before onset) and the share of windows blocked after onset, next to the policy rows.
Alternatives: restrict GX to schema and null checks and let range violations feed drift signals (rejected: changes the deployed behavior the specification describes), leave E2 without GX (rejected: misreports the system).
Consequences: E2 takes longer (about 0.5 s per window for the suite); a strong shift can be reported as blocked rather than detected, which is the behavior of the real system.
