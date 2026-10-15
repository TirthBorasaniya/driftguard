# DriftGuard: Real-Time Drift Detection Pipeline

DriftGuard is a streaming network intrusion detection service. It scores network flows from Kafka
in real time, monitors input drift, prediction drift, and performance drift, retrains
asynchronously when drift is confirmed, and promotes a new model only after it beats the current
model on live traffic in shadow mode and survives a canary stage. Every model change is reversible
with one command. It runs on CICIDS2017 (DistriNet improved release, abrupt drift from new attack
families) and Kyoto 2006+ (AnoShift subsets, gradual drift over ten years).

## Headline results

All numbers below are generated from `results/*.json` by `make results`. A row reading "not yet
run" has no measured value yet.

<!-- results:headline:start -->
| Guarantee | Measured result | Evidence |
|---|---|---|
| Sustained ingestion and p99 latency, with and without retraining | not yet run (`make loadtest`) | `results/e3_load.json` |
| Precision and recall at target recall, time-ordered | not yet run (`make eval-e1`) | `results/e1_model_quality.json` |
<!-- results:headline:end -->

## Architecture

Documented in `docs/architecture.md` (added with the services).

## Quickstart

```bash
make setup && make up && make demo
```

## Evaluation

### E1 Model quality (CICIDS2017)

Time-ordered results lead: the model trains on Monday to Wednesday and is tested on Thursday and
Friday, whose attack families never appear in training. Reproduce with `make eval-e1`.

<!-- results:e1_model_quality:start -->
Not yet run. Reproduce with `make eval-e1`.
<!-- results:e1_model_quality:end -->

### E2 Drift detection benchmark

Shifts are injected into stationary base segments at a known onset; every detector and the
combined policy are scored on detection delay, false alarms, and misses. ADWIN depends on
delayed labels, so results are reported at label coverage 1.0 and 0.3. Reproduce with
`make benchmark-drift`.

<!-- results:e2_drift_detection:start -->
Not yet run. Reproduce with `make benchmark-drift`.
<!-- results:e2_drift_detection:end -->

### E3 System load

Fixed-rate steps from 100 to 2,000 events per second, five minutes each, with the orchestrator
idle and with a retraining run forced mid-step, and with stream features on and off. Reproduce
with `make up && make bootstrap && make loadtest`.

<!-- results:e3_load:start -->
Not yet run. Reproduce with `make loadtest`.
<!-- results:e3_load:end -->

### E3b Kubernetes autoscaling

The kind stack (Strimzi, KEDA, kube-prometheus-stack, MLflow, Prefect, and the application pods)
needs about 12 GB of free RAM. `make k8s-up` refuses to start while the Compose stack or another
kind cluster is running.

On a local kind cluster, the load holds at 250 events/s, steps to 2,000 events/s, and drops
back; KEDA scales the consumers on lag. Reproduce with `make k8s-up && make loadtest-k8s`.

<!-- results:e3b_k8s_autoscaling:start -->
Not yet run. Reproduce with `make loadtest-k8s`.
<!-- results:e3b_k8s_autoscaling:end -->

### ONNX Runtime latency

<!-- results:onnx_latency:start -->
Not yet run. Reproduce with `make benchmark-onnx`.
<!-- results:onnx_latency:end -->

### E4 End-to-end adaptation

Kyoto streams 2008 to 2015 twice from the same initial champion: once with retraining paused
(static) and once with the full policy (adaptive). CICIDS2017 streams the full week with the
policy enabled. Reproduce with `make eval-e4` (drives the live stack; several hours).

<!-- results:e4_adaptation:start -->
Not yet run. Reproduce with `make eval-e4`.
<!-- results:e4_adaptation:end -->

### E5 Promotion gate correctness

Three synthetic challengers go through the shadow gate against one champion on the same 3,000
labeled events. Every alias change in a live run is appended to `results/promotions.jsonl`.
Reproduce with `make eval-e5`.

<!-- results:e5_gate:start -->
Not yet run. Reproduce with `make eval-e5`.
<!-- results:e5_gate:end -->

### First version (historical, transcribed)

<!-- results:v1_verified_run:start -->
| Claim (v1, historical) | Value | Source |
|---|---|---|
| Flows processed by the replay (original CICIDS2017, 5 days) | 863,882 | Producer log, final verification run (commit d0f531e) |
| Invalid rows dropped (negative CICFlowMeter values) | 3 | Preprocessing log, final verification run (commit d0f531e) |
| scale_pos_weight computed from training labels | 30.9120 | Training log, final verification run (commit d0f531e) |
| Test PR-AUC (chronological 75/15/10 split) | 0.1625 | Training log, final verification run (commit d0f531e) |
| Test ROC-AUC (chronological 75/15/10 split) | 0.9383 | Training log, final verification run (commit d0f531e) |
| Precision at recall-calibrated threshold 0.0298 | 0.1719 | Training log, final verification run (commit d0f531e) |
| Recall at recall-calibrated threshold 0.0298 (target 0.95) | 0.9555 | Training log, final verification run (commit d0f531e) |
| PSI trigger window and max PSI | window_0501, max PSI 0.2606 | Consumer log, final verification run (commit d0f531e) |
| Retraining flows started / reached model comparison / aborted at validation | 535 / 534 / 0 | Prefect logs, final verification run (commit d0f531e) |
| Promotions in the live run | 0 | Final verification run; deterministic retraining on identical data |
| Consumer throughput with synchronous retraining (msg/s) | about 250 before, about 64 with retraining completing | Final verification run (commit d0f531e) |

_Transcribed, not re-measured: historical numbers from IMPLEMENTATION_REPORT.md (private, not committed), final verification run at commit d0f531e; split: Chronological 75/15/10 percentile split over the five concatenated original CICIDS2017 TrafficLabelling days (Monday to Friday): train rows 0 to P75, test P75 to P90, stream P90 to end. Not the SPEC E1 protocol (Monday to Wednesday train, Thursday and Friday test)._

_Source: `results/v1_verified_run.json`, run `2026-10-15T00-48-08Z_v1_verified_run`, git `98ff03d`, hardware: arm, 10 logical CPUs._
<!-- results:v1_verified_run:end -->

## Limitations

- The first version's numbers above come from a different split than the SPEC E1 protocol and are
  shown only as history.

## Repository layout

```
src/driftguard/   application code
configs/          YAML configs, one file per concern
scripts/          CLI entry points
tests/            unit, integration, e2e
results/          JSON artifacts rendered into this README
```
