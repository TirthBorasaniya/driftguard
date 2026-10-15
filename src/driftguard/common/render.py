"""Render README tables from results/*.json between `<!-- results:<name>:start/end -->` markers."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from driftguard.common.results import load_results

# ============= Constants =============

MARKER_PATTERN = re.compile(
    r"(<!-- results:(?P<name>[a-z0-9_]+):start -->)(?P<body>.*?)(<!-- results:(?P=name):end -->)",
    re.DOTALL,
)

Renderer = Callable[[dict[str, Any]], str]


# ============= Formatting helpers =============


def fmt(value: Any, digits: int = 4) -> str:
    """Format a number for a markdown table; None renders as `n/a`."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def table(headers: list[str], rows: list[list[Any]], digits: int = 4) -> str:
    """Build a GitHub-flavored markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(cell, digits) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def provenance(doc: dict[str, Any]) -> str:
    """One-line provenance footer for a rendered table."""
    return (
        f"_Source: `results/{doc['run_id'].split('_', 1)[1]}.json`, run `{doc['run_id']}`, "
        f"git `{doc['git_sha']}`, hardware: {doc['environment'].get('hardware', 'n/a')}._"
    )


# ============= Renderers =============


def render_v1_verified_run(doc: dict[str, Any]) -> str:
    """Render the transcribed first-version numbers with their provenance."""
    metrics = doc["metrics"]
    rows = [[item["claim"], item["value"], item["source"]] for item in metrics["items"]]
    return "\n".join(
        [
            table(["Claim (v1, historical)", "Value", "Source"], rows),
            "",
            f"_Transcribed, not re-measured: {doc['notes']}_",
        ]
    )


PROTOCOL_LABELS = {
    "time_ordered": "Time-ordered (train Mon-Wed, test Thu-Fri)",
    "within_day": "Within-day temporal split",
    "random_stratified_upper_bound": "Random stratified (upper-bound reference only)",
}


def _protocol_row(label: str, result: dict[str, Any]) -> list[Any]:
    """One E1 table row from a protocol result."""
    test = result["test"]
    return [
        label,
        test["pr_auc"],
        test["roc_auc"],
        test["precision"],
        test["recall"],
        result["threshold"],
        test["precision_at_recall_0_99"],
        test["n"],
    ]


def render_e1_model_quality(doc: dict[str, Any]) -> str:
    """Render E1: protocols (time-ordered first), per-family recall, and ablations."""
    metrics = doc["metrics"]
    headers = [
        "Protocol",
        "PR-AUC",
        "ROC-AUC",
        "Precision @ thr",
        "Recall @ thr",
        "Threshold",
        "Precision @ recall 0.99",
        "Test rows",
    ]
    rows = [
        _protocol_row(PROTOCOL_LABELS.get(name, name), result)
        for name, result in metrics["protocols"].items()
    ]
    parts = [table(headers, rows)]
    families = metrics["protocols"]["time_ordered"]["test"].get("per_family_recall", {})
    if families:
        parts += [
            "",
            "Per-family recall at the stored threshold (time-ordered test days):",
            "",
            table(
                ["Family", "Recall", "Test rows"],
                [[name, item["recall"], item["n"]] for name, item in families.items()],
            ),
        ]
    ablation_rows = []
    ablations = metrics.get("ablations", {})
    if "o_drop_dst_port" in ablations:
        ablation_rows.append(_protocol_row("Without dst_port", ablations["o_drop_dst_port"]))
    for policy, result in ablations.get("attempted_policy", {}).items():
        ablation_rows.append(_protocol_row(f"attempted_policy={policy}", result))
    stream = ablations.get("stream_features")
    if stream:
        ablation_rows.append(_protocol_row("CICFlowMeter only", stream["cicflowmeter_only"]))
        ablation_rows.append(
            _protocol_row("CICFlowMeter + behavioral", stream["cicflowmeter_plus_behavioral"])
        )
    if ablation_rows:
        parts += ["", "Ablations (time-ordered protocol):", "", table(headers, ablation_rows)]
    return "\n".join(parts)


def _headline_e1(doc: dict[str, Any]) -> str:
    test = doc["metrics"]["protocols"]["time_ordered"]["test"]
    return (
        f"time-ordered PR-AUC {fmt(test['pr_auc'])}, precision {fmt(test['precision'])} "
        f"at recall {fmt(test['recall'])} (target 0.95)"
    )


DETECTOR_ORDER = (
    "psi_max",
    "psi_share",
    "drift_share",
    "score_psi",
    "kswin_score",
    "adwin_error",
    "policy",
)


def render_e2_drift_detection(doc: dict[str, Any]) -> str:
    """Render E2: one table per dataset and label coverage, rows per shift and detector."""
    parts = []
    headers = [
        "Shift",
        "Detector",
        "Median delay (windows)",
        "IQR",
        "Miss rate",
        "False alarms / 100 windows",
    ]
    for dataset, block in doc["metrics"]["datasets"].items():
        for coverage, shifts in block["results"].items():
            rows = []
            for shift, result in shifts.items():
                for name in DETECTOR_ORDER:
                    item = result["detectors"].get(name)
                    if item is None:
                        continue
                    iqr = item["iqr_delay_windows"]
                    rows.append(
                        [
                            shift,
                            name,
                            item["median_delay_windows"],
                            f"{iqr[0]:g} to {iqr[1]:g}" if iqr else "n/a",
                            item["miss_rate"],
                            item["false_alarms_per_100_windows"],
                        ]
                    )
            parts += [f"**{dataset}, {coverage}**", "", table(headers, rows, digits=2), ""]
    return "\n".join(parts).rstrip()


def _headline_e2(doc: dict[str, Any]) -> str:
    items = []
    for dataset, block in doc["metrics"]["datasets"].items():
        full = block["results"].get("label_coverage=1.0", {})
        cov = next((v for k, v in full.items() if k.startswith("covariate:factor=1.5")), None)
        if cov:
            policy = cov["detectors"]["policy"]
            items.append(
                f"{dataset}: policy median delay {fmt(policy['median_delay_windows'], 1)} "
                f"windows, {fmt(policy['false_alarms_per_100_windows'], 2)} false alarms "
                "per 100 windows (covariate x1.5)"
            )
    return "; ".join(items) or "see table"


def render_e5_gate(doc: dict[str, Any]) -> str:
    """Render E5: decision, delta, and interval per synthetic challenger."""
    m = doc["metrics"]
    rows = [
        [
            name,
            r["expected"],
            r["decision"],
            r["delta_pr_auc"],
            f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]",
            r["matches_expected"],
        ]
        for name, r in m["challengers"].items()
    ]
    head = (
        f"{m['shadow_events']:,} shadow events ({m['shadow_positive']:,} attacks), "
        f"margin {next(iter(m['challengers'].values()))['margin']}, "
        f"{next(iter(m['challengers'].values()))['n_resamples']:,} paired resamples."
    )
    return (
        head
        + "\n\n"
        + table(["Challenger", "Expected", "Decision", "Delta PR-AUC", "95% CI", "Matches"], rows)
    )


def _headline_e5(doc: dict[str, Any]) -> str:
    decisions = ", ".join(f"{k}: {v['decision']}" for k, v in doc["metrics"]["challengers"].items())
    return f"{decisions} (all as expected: {fmt(doc['metrics']['all_match_expected'])})"


def render_e3_load(doc: dict[str, Any]) -> str:
    """Render E3: one row per run and rate step."""
    m = doc["metrics"]
    headers = [
        "Run",
        "Target eps",
        "Consumed eps",
        "Lag max",
        "Lag p95",
        "p50 ms",
        "p95 ms",
        "p99 ms",
        "DLQ rate",
        "Redis peak MiB",
        "Saturated",
    ]
    rows = []
    for run, block in m["runs"].items():
        for step in block["steps"]:
            redis_peak = step["redis_peak_used_memory_bytes"]
            rows.append(
                [
                    run,
                    step["target_rate_eps"],
                    step["achieved_consumption_rate_eps"],
                    step["lag"]["max"],
                    step["lag"]["p95"],
                    step["latency_ms"]["p50"],
                    step["latency_ms"]["p95"],
                    step["latency_ms"]["p99"],
                    step["dlq_rate"],
                    redis_peak / 1024**2 if redis_peak else None,
                    step["saturated"],
                ]
            )
    saturation = "; ".join(
        f"{run}: {fmt(block['saturation_rate_eps'])}" for run, block in m["runs"].items()
    )
    return (
        f"{m['partitions']} partitions per topic. Saturation point (first rate whose lag "
        f"grows for the whole step): {saturation}.\n\n" + table(headers, rows, digits=1)
    )


def _headline_e3(doc: dict[str, Any]) -> str:
    runs = doc["metrics"]["runs"]
    parts = []
    for name in ("stream_features=on/idle", "stream_features=on/retraining"):
        steps = runs.get(name, {}).get("steps", [])
        sustained = [s for s in steps if not s["saturated"]]
        if sustained:
            top = sustained[-1]
            parts.append(
                f"{name.split('/')[1]}: {fmt(top['achieved_consumption_rate_eps'], 0)} "
                f"eps sustained, p99 {fmt(top['latency_ms']['p99'], 1)} ms"
            )
    return "; ".join(parts) or "see table"


def render_onnx_latency(doc: dict[str, Any]) -> str:
    """Render the ONNX Runtime versus native LightGBM latency table."""
    rows = [
        [
            r["batch_size"],
            r["native"]["p50_ms"],
            r["onnx"]["p50_ms"],
            r["native"]["p95_ms"],
            r["onnx"]["p95_ms"],
            r["speedup_p50"],
        ]
        for r in doc["metrics"]["rows"]
    ]
    return table(
        ["Batch", "Native p50 ms", "ONNX p50 ms", "Native p95 ms", "ONNX p95 ms", "Speedup (p50)"],
        rows,
    )


def render_e4_adaptation(doc: dict[str, Any]) -> str:
    """Render E4: Kyoto PR-AUC per year (static vs adaptive) and lifecycle counts per run."""
    m = doc["metrics"]
    parts = []
    kyoto = m.get("kyoto")
    if kyoto:
        modes = list(kyoto["modes"])
        years = sorted({y for r in kyoto["modes"].values() for y in r["pr_auc_per_year"]})
        rows = [
            [year]
            + [
                kyoto["modes"][mode]["pr_auc_per_year"].get(year, {}).get("pr_auc")
                for mode in modes
            ]
            for year in years
        ]
        parts += [
            "Kyoto 2006+, PR-AUC per streamed year:",
            "",
            table(["Year"] + [f"{mode} PR-AUC" for mode in modes], rows),
            "",
        ]
        if kyoto.get("figure"):
            parts += [f"![Kyoto PR-AUC per year]({kyoto['figure']})", ""]
    rows = []
    for dataset, block in m.items():
        for mode, result in block["modes"].items():
            life = result["lifecycle"]
            rows.append(
                [
                    dataset,
                    mode,
                    life["drift_events"],
                    life["promoted_to_canary"],
                    life["promotions"],
                    life["declines"],
                    life["rollbacks"],
                ]
            )
    parts += [
        "Model lifecycle per run:",
        "",
        table(
            ["Dataset", "Mode", "Drift events", "To canary", "Promotions", "Declines", "Rollbacks"],
            rows,
        ),
    ]
    cic = m.get("cicids2017")
    if cic and cic.get("figure"):
        parts += ["", f"![CICIDS2017 rolling PR-AUC timeline]({cic['figure']})"]
    return "\n".join(parts)


def _headline_e4(doc: dict[str, Any]) -> str:
    kyoto = doc["metrics"].get("kyoto")
    if not kyoto or not {"static", "adaptive"} <= set(kyoto["modes"]):
        return "see E4 section"

    def mean(mode: str) -> float | None:
        values = [
            v["pr_auc"]
            for v in kyoto["modes"][mode]["pr_auc_per_year"].values()
            if v["pr_auc"] is not None
        ]
        return sum(values) / len(values) if values else None

    return (
        f"Kyoto mean yearly PR-AUC: adaptive {fmt(mean('adaptive'))} versus static "
        f"{fmt(mean('static'))}"
    )


def render_e3b(doc: dict[str, Any]) -> str:
    """Render E3b: scaling timings and the duplicate and loss check."""
    m = doc["metrics"]
    s, c = m["scaling"], m["duplicate_and_loss_check"]
    rows = [
        [
            "Time from load step to first new consumer pod Ready (s)",
            s["time_to_first_new_pod_ready_s"],
        ],
        ["Peak consumer lag (messages)", s["peak_lag"]],
        ["Maximum replicas", s["max_replicas"]],
        [
            "Time to drain lag below 5,000 after the load drop (s)",
            s["time_to_drain_below_threshold_s"],
        ],
        ["Scale-in after the load drop (s)", s["scale_in_after_load_drop_s"]],
        ["Distinct event ids produced", c["produced_distinct"]],
        ["Distinct event ids predicted", c["predicted_distinct"]],
        ["Duplicate predictions", c["duplicates"]],
        ["Lost events", c["lost"]],
    ]
    phases = ", ".join(f"{p['rate_eps']:g} eps for {p['duration_s']:g} s" for p in m["phases"])
    out = f"kind, phases: {phases}.\n\n" + table(["Measure", "Value"], rows, digits=1)
    if m.get("figure"):
        out += f"\n\n![Consumer replicas over time]({m['figure']})"
    return out


def _headline_e3b(doc: dict[str, Any]) -> str:
    m = doc["metrics"]
    c = m["duplicate_and_loss_check"]
    return (
        f"1 to {m['scaling']['max_replicas']} consumers on lag, first new pod in "
        f"{fmt(m['scaling']['time_to_first_new_pod_ready_s'], 0)} s, "
        f"{fmt(c['duplicates'])} duplicates, {fmt(c['lost'])} lost (kind)"
    )


RENDERERS: dict[str, tuple[str, Renderer | None]] = {
    "v1_verified_run": ("make import-v1-results", render_v1_verified_run),
    "e1_model_quality": ("make eval-e1", render_e1_model_quality),
    "e2_drift_detection": ("make benchmark-drift", render_e2_drift_detection),
    "e5_gate": ("make eval-e5", render_e5_gate),
    "e3_load": ("make loadtest", render_e3_load),
    "onnx_latency": ("make benchmark-onnx", render_onnx_latency),
    "e4_adaptation": ("make eval-e4", render_e4_adaptation),
    "e3b_k8s_autoscaling": ("make loadtest-k8s", render_e3b),
}

# each headline row: (guarantee, results name, make target, extractor returning the value text)
HEADLINE_ROWS: list[tuple[str, str, str, Callable[[dict[str, Any]], str]]] = [
    (
        "Sustained ingestion and p99 latency, with and without retraining",
        "e3_load",
        "make loadtest",
        _headline_e3,
    ),
    (
        "Precision and recall at target recall, time-ordered",
        "e1_model_quality",
        "make eval-e1",
        _headline_e1,
    ),
]


def render_headline(results_dir: Path) -> str:
    """Render the headline table, one row per SPEC guarantee with its evidence file."""
    rows = []
    for claim, name, make_target, extractor in HEADLINE_ROWS:
        doc = load_results(name, results_dir)
        value = extractor(doc) if doc is not None else f"not yet run (`{make_target}`)"
        rows.append([claim, value, f"`results/{name}.json`"])
    if not rows:
        return "No experiments registered."
    return table(["Guarantee", "Measured result", "Evidence"], rows)


def register(name: str, make_target: str, renderer: Renderer) -> None:
    """Register a renderer for a results name."""
    RENDERERS[name] = (make_target, renderer)


# ============= Rewriting =============


def render_block(name: str, results_dir: Path) -> str:
    """
    Render the markdown body for one marker.

    Parameters
    ----------
    name : str
        Results name.
    results_dir : Path
        Directory with results JSON files.

    Returns
    -------
    body : str
        Markdown placed between the markers.
    """
    make_target, renderer = RENDERERS.get(name, (f"make results ({name})", None))
    if name == "headline":
        return render_headline(results_dir)
    doc = load_results(name, results_dir)
    if doc is None or renderer is None:
        return f"Not yet run. Reproduce with `{make_target}`."
    return renderer(doc) + "\n\n" + provenance(doc)


def render_readme(readme_text: str, results_dir: Path) -> str:
    """Return README text with every results marker block re-rendered."""

    def _sub(match: re.Match) -> str:
        body = render_block(match.group("name"), results_dir)
        return f"{match.group(1)}\n{body}\n{match.group(4)}"

    return MARKER_PATTERN.sub(_sub, readme_text)
