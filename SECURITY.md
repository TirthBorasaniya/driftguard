# Security Notes

This is a portfolio and demonstration project, not a deployed service. It is
published so the engineering can be read. This document records the security
posture honestly, including the parts that are not fully resolved.

## Threat model

DriftGuard runs as a local Docker Compose stack. There is no multi-tenancy, no
user accounts, and no untrusted input path: the only data it ingests is
CICIDS2017 CSV files that the operator downloads and places on disk, replayed
through Kafka by a local producer. Nothing in this stack is intended to be
remotely reachable.

## Network exposure

Every published port in `docker-compose.yml` is bound to `127.0.0.1`, not
Docker's default of `0.0.0.0`.

| Service | Port | Authentication |
|---|---|---|
| api | 8000 | none (read-only demo) |
| onnx-server | 8001 | none (read-only demo) |
| kafka | 9092 | none (PLAINTEXT) |
| schema-registry | 8081 | none |
| redis | 6379 | none |
| prometheus | 9090 | none |
| grafana | 3000 | admin password required via `GF_ADMIN_PASSWORD` |

Redis, Kafka, Schema Registry, and Prometheus have no authentication in this
stack. Unauthenticated Redis reachable on all interfaces is an actively
exploited pattern (module-load and cron-write remote code execution), which is
why the loopback binding matters and should not be reverted. Do not publish
these ports without adding authentication first.

`GF_ADMIN_PASSWORD` has no default. Compose fails loudly via
`${GF_ADMIN_PASSWORD:?...}` if it is unset, rather than booting Grafana with a
guessable password.

## API endpoints

The serving API is intentionally unauthenticated: it is a read-only demo. The
payloads, not the access control, are what were hardened.

`GET /predictions/recent` previously did `SELECT *`, which returned two columns
that carry network identifiers:

- `features_json`, which persists the submitted `src_ip`
- `event_id`, which defaults to the caller-supplied `flow_id`, and whose
  CICIDS2017 form is the full five-tuple
  `src_ip-dst_ip-src_port-dst_port-protocol`

Both are now excluded by an explicit `response_model` (`RecentPredictionItem`),
and the query selects named columns so that adding a column to the
`predictions` table cannot silently widen the public payload.
`GET /predictions/stats` was reviewed and returns aggregates only, with no
identifiers, no per-event rows, and no filesystem or infrastructure detail.

There is no endpoint that executes shell commands, evaluates code, reads
arbitrary file paths, or fetches a caller-supplied URL. The two `subprocess`
call sites (`src/features/materializer.py`) use list-form arguments with no
`shell=True` and no caller-controlled input, and are not reachable from any
HTTP route.

## MLflow: 43 advisories, and why nearly all are unreachable here

`pip-audit` reports 43 advisories against the pinned `mlflow==2.12.2`. This is
the largest single dependency finding in the project, so it is worth being
precise about which parts apply.

**Nearly all of them require an exposed MLflow tracking server.** The
advisories concentrate in server-side surfaces: `/ajax-api/3.0/jobs/*`
authorization gaps, `/graphql` denial of service, `--serve-artifacts` multipart
upload authorization, the `basic-auth` app's `BEFORE_REQUEST_HANDLER` checks,
artifact-handler and `_create_model_version` directory traversal, and
`_create_webhook` server-side request forgery.

**This project runs no MLflow server.** The tracking backend is a local SQLite
file (`MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db`), there is no `mlflow`
service in `docker-compose.yml`, and no MLflow port is published. `mlflow ui`
exists only as an optional local command for browsing runs. With no server
listening, the server-side advisories have no reachable attack surface.

### The one that is reachable: CVE-2024-37056

`CVE-2024-37056` is deserialization of untrusted data in MLflow's model
loading, specifically reachable through the LightGBM/scikit-learn flavor: a
maliciously crafted model artifact can execute arbitrary code when loaded.
This project calls `mlflow.lightgbm.load_model()` in four places
(`src/serving/model_loader.py`, `src/orchestration/flows/retraining_flow.py`,
`src/serving/onnx_export.py`).

**There is no patch. The advisory lists no fix version.** This is not something
a version bump resolves.

What limits it here:

- Models are loaded only from the local MLflow registry backed by a SQLite
  file on the same machine, and from `models/production_model.pkl` in this
  repository. There is no path by which a remote party supplies a model.
- The local fallback loader uses `joblib.load`, which is pickle-based and
  carries the same class of risk. Same conclusion: the input is a local,
  trusted, in-repo artifact.

The residual risk is a supply-chain one rather than a runtime one: because
`models/production_model.pkl` is committed (Render deploys from it), a merged
malicious pull request that swapped that file would execute code on anyone who
subsequently ran the API. Review changes to files under `models/` with the
same care as changes to code.

## Dependency posture

`pip-audit` reports 97 advisories across 17 packages. Assessed for reachability
in this codebase rather than by version match alone:

- **lightgbm** — was 4.3.0, carrying `CVE-2024-43598` (remote code execution).
  This was the one finding genuinely reachable in a core code path, and it is
  **fixed**: pinned to 4.6.0. Verified the committed model still loads and
  predicts identically and that ONNX conversion still succeeds.
- **setuptools** — was pinned `<70`, which blocked the fixes for
  `CVE-2024-6345` (package_index RCE, fixed in 70.0.0) and `CVE-2025-47273`
  (path traversal, fixed in 78.1.1). Now `>=78.1.1,<81`. The upper bound is
  required, not cosmetic: mlflow 2.12.2 imports `pkg_resources`, which
  setuptools removed in 81.0.0.
- **starlette** — 9 advisories, **none reachable**. Every form-parsing denial
  of service requires `request.form()`; the StaticFiles SSRF and HTTPEndpoint
  advisories require those constructs. This codebase uses none of them (JSON
  request bodies only, via Pydantic). Patch hygiene, not exposure.
- **pyarrow** — `CVE-2024-52338` affects the Arrow **R** package, not Python.
  `CVE-2026-25087` requires reading Arrow **IPC** files; this project reads
  Parquet, from locally generated paths.
- **feast** — `CVE-2025-11157` is in the Kubernetes materializer, which this
  project does not use (local file offline store, Redis online store).
- **gitpython, jupyterlab, pyasn1, protobuf, nltk, click** — transitive, not
  declared in `requirements.txt`. `click.edit()`, the sink for
  `CVE-2026-7246`, is never called.

## Secrets

No credential has ever been committed. Verified across all commits on all refs:
no `.env`, no private keys or certificates, no cloud credential files, and no
matches for common token prefixes. The only credential-shaped string in history
is the literal placeholder `changeme`, which was a Grafana fallback default and
has since been replaced with a required variable.

`.env` is gitignored and untracked. `.env.example` ships only non-secret
localhost defaults with an empty `GF_ADMIN_PASSWORD`.

Note that a prior commit message (`559c203`) claimed to have removed roughly
64 MB of committed dataset records and stated that they "remain in prior
history." That claim is false; those paths never existed in any commit. See
commit `467f2a5` for the correction. No history rewrite is needed.

## Reporting

This is a portfolio project with no deployment and no users. If you find
something wrong, please open an issue.
