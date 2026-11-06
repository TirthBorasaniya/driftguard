# Runbook

## Start and stop

| Task | Command |
|---|---|
| Start the Compose stack | `make up` (downloads the data and fits GX bounds if missing) |
| Create the initial champion | `make bootstrap` |
| Replay a stream | `make replay ARGS="--mode fixed_rate --rate 500"` |
| Follow logs | `make logs` |
| Stop | `make down` |
| Kubernetes | `make k8s-up`, `make k8s-down` |

Only one stack runs at a time: `make up` and `make k8s-up` refuse to start while another Compose
project or kind cluster is running.

## Roll back a model

```bash
make rollback DATASET=cicids2017
```

This calls `POST /admin/rollback`, which moves `previous_champion` to `champion` and returns to
`champion_only`. The serving watcher swaps sessions within 10 s. The change is appended to
`results/promotions.jsonl`. It fails with a conflict when no previous champion exists.

## Pause or resume retraining

```bash
docker compose exec redis redis-cli set dg:retrain:paused:cicids2017 operator
docker compose exec redis redis-cli del dg:retrain:paused:cicids2017
```

A data-quality pause is set by the monitor and lifted by the next clean window; an operator pause
is never lifted automatically.

## Alerts

| Alert | Meaning | First action |
|---|---|---|
| `ConsumerLagHigh` | Lag above 50,000 for 2 min | Check serving latency; on Kubernetes check the ScaledObject |
| `DLQRateHigh` | More than 1% of events dead-lettered | Inspect `error_type` headers on `flows.dlq` |
| `ScoringUnavailable` | Batches dead-lettered after retries | Check `serving` health and the champion alias |
| `DriftDetected` | HIGH severity drift event | Watch the retraining run and the gate decision |
| `RetrainingFailed` | A run failed or could not be submitted | Read the Prefect run logs |
| `DataQualityFailed` | A window failed the GX suite, retraining paused | Inspect the failed expectations in the monitor log |
| `CanaryRolledBack` | A canary or champion was rolled back | Read the breaches in `promotions.jsonl` |
| `ScoringLatencyP95Breach`, `ScoringLatencyP99Breach` | Latency above the SLO | Check CPU and batch size |

## Replay dead letters

Messages on `flows.dlq` keep their original bytes and carry `error_type`, `error_msg`,
`source_topic`, and `source_offset`. Fix the cause, then re-produce the original value to the
source topic; idempotency keys prevent double scoring of events that were already scored.

## Refit nothing to make a run pass

The percentile GX bounds are fitted once (`make fit-gx-bounds`) and only feed the reported
`gx_percentile_failure` signal; never refit them to change that signal. Training validation uses
physical bounds (DECISIONS D-022). A rejected retraining run (`rejected_data_quality`) means the
window holds null, infinite, or physically impossible values; do not loosen those limits.
`make report-gx-percentile` counts the real windows that fall outside the percentile bounds.

## Troubleshooting

- `serving` not ready: no champion is registered. Run `make bootstrap`.
- Monitor logs `percentile_bounds_missing` and reports no percentile signal:
  `configs/data/<dataset>_gx_bounds.json` is missing. Run `make fit-gx-bounds`.
- `PermissionError` writing `/app/results` or `/app/data`: the image was built for another uid.
  The image runs as `HOST_UID` and `HOST_GID` (`make` exports yours). On Linux, when calling
  `docker compose` directly, run `HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose up -d --build`.
  Never run compose as root, and never `chmod 777` the mounts (DECISIONS D-026).
- Stream features fail at startup: `IP_HASH_SALT` is empty.
- Prefect worker cannot deploy: check that `prefect-server` is healthy and `PREFECT_API_URL` is set.
