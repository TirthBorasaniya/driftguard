"""E5 gate correctness: writes results/e5_gate.json (SPEC 9, E5)."""

import argparse
from pathlib import Path

from driftguard.benchmark.gate_eval import run_e5
from driftguard.common.logging import configure_logging
from driftguard.common.results import write_results
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "e5.yaml"
NOTES = (
    "Champion trained on Monday and Tuesday. Challengers: better (adds the older half of "
    "Wednesday, the post-drift day), equal (same data and seed), worse (shuffled labels). All are "
    "compared on the same 3,000 events sampled from the newer half of Wednesday with 1,000 "
    "paired bootstrap resamples. Attack campaigns span both halves of the day, so the better "
    "challenger's near-perfect PR-AUC reflects that overlap."
)


def main() -> None:
    """Run E5 and write its results file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    configure_logging("eval-e5")
    metrics = run_e5(str(args.config))
    print(f"Wrote {write_results('e5_gate', metrics, args.config, notes=NOTES)}")


if __name__ == "__main__":
    main()
