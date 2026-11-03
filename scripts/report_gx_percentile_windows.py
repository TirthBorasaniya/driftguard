"""Count real windows outside the percentile bounds; writes results/gx_percentile_windows.json."""

import argparse
from pathlib import Path

from driftguard.benchmark.gx_percentile import run_gx_percentile
from driftguard.common.logging import configure_logging
from driftguard.common.results import write_results
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "gx_percentile.yaml"
NOTES = (
    "Counts, per dataset, the windows of the real stream that fail the Great Expectations suite on "
    "the percentile bounds fitted once on the initial reference data (0.1 and 99.9 percentiles, "
    "at least `mostly` of values inside). These are the windows the percentile suite would have "
    "blocked had it gated retraining; since D-022 it is a reported signal only. Windows with "
    "`in_reference_period` true end inside the initial training days."
)


def main() -> None:
    """Run the scan and write its results file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    configure_logging("report-gx-percentile")
    metrics = run_gx_percentile(str(args.config.relative_to(REPO_ROOT)))
    print(f"Wrote {write_results('gx_percentile_windows', metrics, args.config, notes=NOTES)}")


if __name__ == "__main__":
    main()
