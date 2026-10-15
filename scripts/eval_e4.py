"""E4 end-to-end adaptation on the live stack: writes results/e4_adaptation.json and figures."""

import argparse
from pathlib import Path

from driftguard.benchmark.adaptation import run_dataset
from driftguard.benchmark.figures import cicids_timeline, kyoto_per_year
from driftguard.common.config import load_config
from driftguard.common.logging import configure_logging
from driftguard.common.results import write_results
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "e4.yaml"
NOTES = (
    "Live Compose stack. Kyoto: initial champion trained on 2006 and 2007, stream 2008 to 2015 "
    "replayed twice from the same initial champion, static (retraining paused by an operator "
    "flag) and adaptive (full policy). CICIDS2017: initial champion on Monday and Tuesday, full "
    "week streamed with the policy enabled. PR-AUC is computed on the served score."
)


def main() -> None:
    """Run E4 for both datasets, write figures and the results file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", action="append", dest="datasets")
    args = parser.parse_args()
    configure_logging("eval-e4")
    config = load_config(args.config).data
    figures = REPO_ROOT / config["figures_dir"]
    metrics = {}
    for dataset in args.datasets or ["kyoto", "cicids2017"]:
        metrics[dataset] = run_dataset(dataset, config)
    if "kyoto" in metrics:
        per_year = {m: r["pr_auc_per_year"] for m, r in metrics["kyoto"]["modes"].items()}
        metrics["kyoto"]["figure"] = str(
            kyoto_per_year(per_year, figures / "e4_kyoto.png").relative_to(REPO_ROOT)
        )
    if "cicids2017" in metrics:
        adaptive = metrics["cicids2017"]["modes"]["adaptive"]
        timeline = adaptive["timeline"]
        drift_times = [e["window_end_ms"] for e in timeline["drift_events"] if e["window_end_ms"]]
        gate = [
            (g["event_time_ms"], g["action"])
            for g in timeline["gate_decisions"]
            if g.get("event_time_ms")
        ]
        metrics["cicids2017"]["figure"] = str(
            cicids_timeline(
                adaptive["rolling_pr_auc"], drift_times, gate, figures / "e4_cicids2017.png"
            ).relative_to(REPO_ROOT)
        )
    print(f"Wrote {write_results('e4_adaptation', metrics, args.config, notes=NOTES)}")


if __name__ == "__main__":
    main()
