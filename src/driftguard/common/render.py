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


RENDERERS: dict[str, tuple[str, Renderer | None]] = {
    "v1_verified_run": ("make import-v1-results", render_v1_verified_run),
    "e1_model_quality": ("make eval-e1", render_e1_model_quality),
}

# each headline row: (guarantee, results name, make target, extractor returning the value text)
HEADLINE_ROWS: list[tuple[str, str, str, Callable[[dict[str, Any]], str]]] = [
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
