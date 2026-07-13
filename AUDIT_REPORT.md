# Audit Report: ONNX Export and Inference-Serving Layer

Mandatory pre-implementation audit for adding ONNX export, an ONNX Runtime FastAPI
server, and a native-vs-ONNX benchmark to DriftGuard. No code was written before this
report was produced.

## 0. CLAUDE.md domain discrepancy (important)

`CLAUDE.md` at the repo root still documents the **original fraud-detection domain**
(Sparkov credit-card transaction dataset, `TransactionEvent` schema, `cc_num` entity,
`scale_pos_weight=172`, `transactions.dlq` topic, F2 operating point). This does not
match the actual current codebase, which was migrated in a prior session to **network
telemetry anomaly detection on CICIDS2017** (confirmed via `IMPLEMENTATION_REPORT.md`,
`src/config.py`, and the full commit history). `CLAUDE_domain_migration.md` (gitignored,
present locally) documents this migration and is the source of truth over `CLAUDE.md`
wherever they conflict, per its own header.

**This audit and all subsequent implementation work uses the real, current codebase
state** (network anomaly detection, `network-anomaly-detector` model, 10 numeric
`FEATURE_COLS`), not the stale fraud-detection description in `CLAUDE.md`. Naming/style
rules from `CLAUDE.md`'s Code Style section still apply and are followed (they are
domain-agnostic).

## 1. Directory tree (two levels, noise directories excluded)

```
.
├── .env
├── .env.example
├── AUDIT_REPORT.md
├── benchmark/
│   └── locustfile.py
├── CLAUDE_domain_migration.md
├── CLAUDE.md
├── docker-compose.yml
├── Dockerfile.api
├── IMPLEMENTATION_REPORT.md
├── Makefile
├── models/
│   ├── .gitkeep
│   ├── champion_metrics.json
│   ├── feature_cols.json
│   ├── production_model.pkl
│   └── threshold.json
├── monitoring/
│   ├── dashboards/
│   ├── grafana/
│   ├── prometheus/
│   └── prometheus.yml
├── notebooks/
├── P1_COMBINED_SESSION.md
├── packages.txt
├── pyproject.toml
├── README.md
├── render.yaml
├── requirements.txt
├── scripts/
│   └── generate_reference_dataset.py
├── src/
│   ├── config.py
│   ├── consumer/
│   ├── data/
│   ├── features/
│   ├── monitoring/
│   ├── orchestration/
│   ├── producer/
│   ├── schemas/
│   ├── serving/
│   ├── training/
│   └── validation/
└── tests/
    └── (22 test files)
```

(`data/`, `mlruns/`, `reports/`, `.venv311/`, and cache directories omitted as noise.)

## 2. `src/config.py` — printed in full

See the file directly; key points relevant to this task:
- `Settings.mlflow_tracking_uri = "sqlite:///mlruns/mlflow.db"`
- `Settings.mlflow_model_name = "network-anomaly-detector"`
- `Settings.mlflow_champion_alias = "champion"`
- `FEATURE_COLS` imported from `src.features.engineering` (10 features, see section 6)
- `MODELS_DIR = PROJECT_ROOT / "models"`

## 3. MLflow tracking URI

Consistent across every location checked:
- `src/config.py`: `mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"` (default)
- `.env`: `MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db`
- `.env.example`: `MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db`
- `render.yaml`: `MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db`
- `docker-compose.yml`: no `MLFLOW_TRACKING_URI` reference (no separate `mlflow` service
  exists in this compose file; the backend is a local SQLite file, not a tracking server)
- Shell environment: `MLFLOW_TRACKING_URI` not set

**This is a file-based SQLite backend, not a running MLflow tracking server.** No
server process needs to be started to load models or query the registry; the Python
client (`mlflow.set_tracking_uri(...)` + `MlflowClient()`) reads the SQLite file
directly. `mlflow ui` (documented in `CLAUDE.md`'s Common Commands) only starts the
web UI, not a requirement for programmatic access.

## 4. Champion model registry lookup

There is no `mlflow models list` CLI command in real MLflow (`mlflow models` only has
`serve`, `predict`, `build-docker`, `generate-dockerfile`, `prepare-env`). Used the
equivalent: `MlflowClient().search_registered_models()` / `get_model_version_by_alias`.

Three registered models exist in this registry:
- `fraud-detector` — champion alias -> version 1 (stale, from the original fraud domain)
- `fraud-detector-ieee` — champion alias -> version 1 (stale, from the original fraud domain)
- **`network-anomaly-detector`** — the real, current model. Champion alias resolves to:
  - **version 2**
  - run_id: `45235b02763645bd9a13ca6c1ab9f6fd`
  - artifact path: `/Users/tirth/driftguard/mlruns/3/45235b02763645bd9a13ca6c1ab9f6fd/artifacts/model`
  - status: `READY`

  Versions 3 through 539 also exist under this model name (challengers registered by
  a prior live-streaming verification run, none promoted — the champion alias has
  remained at version 2 since that run, confirmed by direct trace in
  `IMPLEMENTATION_REPORT.md`'s Final Verification Run section).

**Champion alias exists and resolves cleanly.** No need to stop and report a broken
registry; `mlflow_model_name = "network-anomaly-detector"` and
`mlflow_champion_alias = "champion"` from `src/config.py` are correct and will be used
as-is in the export script.

## 5. Installed package versions

| Package | Version before this audit | Action |
|---|---|---|
| lightgbm | 4.3.0 | already installed |
| fastapi | 0.111.0 | already installed |
| uvicorn | 0.30.0 | already installed |
| onnxmltools | not installed | installed: **1.16.0** |
| skl2onnx | not installed | installed: **1.20.0** |
| onnxruntime | not installed | installed: **1.27.0** |
| onnx (transitive dependency of onnxmltools) | not installed | installed: **1.22.0** |

All three missing packages installed into `.venv311` via `pip install onnxmltools
skl2onnx onnxruntime`. `onnxmltools.convert_lightgbm` and
`onnxmltools.convert.common.data_types.FloatTensorType` both import successfully.

## 6. Feature column list

From `src/config.py` (`FEATURE_COLS`, imported from `src.features.engineering`):

```python
['flow_duration', 'flow_bytes_per_sec', 'flow_packets_per_sec', 'total_fwd_packets',
 'total_bwd_packets', 'packet_length_mean', 'packet_length_std', 'flow_iat_mean',
 'fwd_bwd_packet_ratio', 'syn_flag_count']
```

**Length: 10.** All numeric, no categorical columns (`CATEGORICAL_COLS = []`), so
`initial_types=[("input", FloatTensorType([None, 10]))]` is correct for the ONNX export
with no encoding step needed.

## 7. Canonical dependency file

`requirements.txt` is canonical (pinned `==` versions throughout). `pyproject.toml`
contains only `[build-system]` and tool configuration (`ruff`, `mypy`, `pytest`) — no
dependency list. New packages will be pinned into `requirements.txt`.

## 8. MLflow startup command in CLAUDE.md

Already present under `## Common Commands`:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```
No addition needed — this satisfies the "if none exists, add one" condition (one
already exists), even though, per section 3 above, it is not actually required to load
the champion model programmatically.

## Open discrepancy flagged for the user before Step 6

The task's Step 6 says "Confirm `models/champion.onnx` is in `.gitignore` (model
artifacts are not committed)." This is a **factual mismatch with the existing repo
convention**: `.gitignore` explicitly says `# models/  <- intentionally NOT ignored`,
and `production_model.pkl`, `champion_metrics.json`, `threshold.json`, and
`feature_cols.json` under `models/` are already tracked and committed (confirmed via
git history — `render.yaml` deploys directly from the committed model artifact).
Gitignoring only `champion.onnx` while every other model artifact in the same directory
is committed would be an inconsistent, one-off exception. This will be raised explicitly
before Step 6 rather than silently resolved either way.
