"""Generate the Grafana dashboards under grafana/dashboards/ from code (`--check` verifies them)."""

import argparse
import json
import sys

from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

OUT_DIR = REPO_ROOT / "grafana" / "dashboards"
DATASOURCE = {"type": "prometheus", "uid": "prometheus"}


# ============= Builders =============


def panel(
    panel_id: int,
    title: str,
    exprs: list[tuple[str, str]],
    x: int,
    y: int,
    unit: str = "short",
    kind: str = "timeseries",
    w: int = 12,
    h: int = 8,
) -> dict:
    """One panel with PromQL targets given as (expr, legend) pairs."""
    return {
        "id": panel_id,
        "type": kind,
        "title": title,
        "datasource": DATASOURCE,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "targets": [
            {"refId": chr(65 + i), "expr": expr, "legendFormat": legend, "datasource": DATASOURCE}
            for i, (expr, legend) in enumerate(exprs)
        ],
    }


def dashboard(uid: str, title: str, panels: list[dict]) -> dict:
    """A dashboard with a dataset template variable."""
    return {
        "uid": uid,
        "title": title,
        "tags": ["driftguard"],
        "schemaVersion": 39,
        "version": 1,
        "refresh": "10s",
        "time": {"from": "now-1h", "to": "now"},
        "templating": {
            "list": [
                {
                    "name": "dataset",
                    "type": "query",
                    "datasource": DATASOURCE,
                    "query": "label_values(driftguard_events_consumed_total, dataset)",
                    "refresh": 2,
                    "current": {"text": "cicids2017", "value": "cicids2017"},
                }
            ]
        },
        "panels": panels,
    }


DS = 'dataset="$dataset"'


def pipeline_health() -> dict:
    """Throughput, lag, latency, DLQ, duplicates."""
    return dashboard(
        "dg-pipeline",
        "DriftGuard: Pipeline health",
        [
            panel(
                1,
                "Events consumed per second",
                [(f"sum(rate(driftguard_events_consumed_total{{{DS}}}[1m]))", "consumed")],
                0,
                0,
                "ops",
            ),
            panel(
                2,
                "Consumer lag (messages)",
                [(f"sum by (partition) (driftguard_consumer_lag{{{DS}}})", "p{{partition}}")],
                12,
                0,
            ),
            panel(
                3,
                "Scoring latency per batch",
                [
                    (
                        f"histogram_quantile({q}, sum by (le) "
                        f"(rate(driftguard_scoring_latency_seconds_"
                        f'bucket{{{DS},job="serving"}}[5m])))',
                        f"p{int(q * 100)}",
                    )
                    for q in (0.5, 0.95, 0.99)
                ],
                0,
                8,
                "s",
            ),
            panel(
                4,
                "DLQ and duplicates per second",
                [
                    (
                        f"sum by (error_type) (rate(driftguard_events_dlq_total{{{DS}}}[5m]))",
                        "dlq {{error_type}}",
                    ),
                    (f"sum(rate(driftguard_duplicates_total{{{DS}}}[5m]))", "duplicates"),
                ],
                12,
                8,
                "ops",
            ),
            panel(
                5,
                "Stream feature time per batch",
                [
                    (
                        f"histogram_quantile(0.99, sum by (le) "
                        f"(rate(driftguard_stream_feature_seconds_"
                        f"bucket{{{DS}}}[5m])))",
                        "p99",
                    )
                ],
                0,
                16,
                "s",
            ),
            panel(
                6,
                "Late events per second",
                [(f"sum(rate(driftguard_late_events_total{{{DS}}}[5m]))", "late")],
                12,
                16,
                "ops",
            ),
        ],
    )


def drift_monitor() -> dict:
    """Drift signals, detector firings, drift events, data quality."""
    return dashboard(
        "dg-drift",
        "DriftGuard: Drift monitor",
        [
            panel(
                1,
                "Feature PSI",
                [
                    (f"driftguard_psi_max{{{DS}}}", "psi_max"),
                    (f"driftguard_psi_share{{{DS}}}", "psi_share"),
                ],
                0,
                0,
            ),
            panel(
                2,
                "Evidently drift share and score PSI",
                [
                    (f"driftguard_drift_share{{{DS}}}", "drift_share"),
                    (f"driftguard_score_psi{{{DS}}}", "score_psi"),
                ],
                12,
                0,
            ),
            panel(
                3,
                "Change detector firings",
                [
                    (f"increase(driftguard_adwin_detections_total{{{DS}}}[5m])", "ADWIN"),
                    (f"increase(driftguard_kswin_detections_total{{{DS}}}[5m])", "KSWIN"),
                ],
                0,
                8,
            ),
            panel(
                4,
                "Drift events",
                [(f"increase(driftguard_drift_events_total{{{DS}}}[5m])", "{{severity}}")],
                12,
                8,
            ),
            panel(
                5,
                "Windows closed and flagged",
                [
                    (f"increase(driftguard_windows_closed_total{{{DS}}}[5m])", "closed"),
                    (f"increase(driftguard_windows_flagged_total{{{DS}}}[5m])", "flagged"),
                ],
                0,
                16,
            ),
            panel(
                6,
                "Data quality and pause",
                [
                    (
                        f"increase(driftguard_data_quality_failures_total{{{DS}}}[10m])",
                        "GX failures",
                    ),
                    (f"driftguard_retraining_paused{{{DS}}}", "paused"),
                    (f"driftguard_labeled_since_promotion{{{DS}}}", "labels since promotion"),
                ],
                12,
                16,
            ),
        ],
    )


def model_lifecycle() -> dict:
    """Served versions, rollout mode, retraining, promotions, shadow agreement."""
    return dashboard(
        "dg-lifecycle",
        "DriftGuard: Model lifecycle",
        [
            panel(
                1,
                "Served model versions",
                [(f"driftguard_model_info{{{DS}}} == 1", "{{alias}} v{{version}}")],
                0,
                0,
                kind="table",
            ),
            panel(
                2,
                "Rollout mode",
                [(f"driftguard_rollout_mode{{{DS}}} == 1", "{{mode}}")],
                12,
                0,
                kind="stat",
            ),
            panel(
                3,
                "Retraining triggers and runs",
                [
                    (
                        f"increase(driftguard_retraining_triggers_total{{{DS}}}[15m])",
                        "trigger {{result}}",
                    ),
                    (f"increase(driftguard_retraining_runs_total{{{DS}}}[15m])", "run {{status}}"),
                ],
                0,
                8,
            ),
            panel(
                4,
                "Promotion gate outcomes",
                [(f"increase(driftguard_promotions_total{{{DS}}}[1h])", "{{result}}")],
                12,
                8,
            ),
            panel(
                5,
                "Shadow agreement ratio",
                [(f"driftguard_shadow_agreement_ratio{{{DS}}}", "agreement")],
                0,
                16,
                "percentunit",
            ),
            panel(
                6,
                "Positive rate by alias",
                [
                    (
                        f"sum by (alias) (rate(driftguard_positive_decisions_total{{{DS}}}[5m])) / "
                        f"sum by (alias) (rate(driftguard_scored_events_total{{{DS}}}[5m]))",
                        "{{alias}}",
                    )
                ],
                12,
                16,
                "percentunit",
            ),
        ],
    )


DASHBOARDS = {
    "pipeline_health.json": pipeline_health,
    "drift_monitor.json": drift_monitor,
    "model_lifecycle.json": model_lifecycle,
}


# ============= CLI =============


def main() -> int:
    """Write or check the dashboard files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", dest="o_check")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for file_name, builder in DASHBOARDS.items():
        text = json.dumps(builder(), indent=2) + "\n"
        path = OUT_DIR / file_name
        if path.exists() and path.read_text() == text:
            continue
        stale.append(file_name)
        if not args.o_check:
            path.write_text(text)
    if args.o_check and stale:
        print(f"Stale dashboards: {stale}; run scripts/build_dashboards.py")
        return 1
    print("Dashboards up to date" if not stale else f"Wrote {stale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
