"""E2 drift detection benchmark: writes results/e2_drift_detection.json (SPEC 9, E2)."""

import argparse
from pathlib import Path

from driftguard.benchmark.drift_benchmark import run_e2
from driftguard.common.logging import configure_logging
from driftguard.common.results import write_results
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "e2.yaml"
NOTES = (
    "Offline benchmark with the production monitor core (PSI, Evidently, score PSI, KSWIN, ADWIN, "
    "policy v3). Base segments are shuffled, hence stationary; one shift is injected per run at a "
    "random onset. Delay is measured in windows from the onset window; a miss is no detection "
    "within the horizon. False alarms count flagged windows before onset."
)


def main() -> None:
    """Run E2 and write its results file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    configure_logging("benchmark-drift")
    metrics = run_e2(str(args.config))
    print(f"Wrote {write_results('e2_drift_detection', metrics, args.config, notes=NOTES)}")


if __name__ == "__main__":
    main()
