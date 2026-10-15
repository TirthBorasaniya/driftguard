"""E3 system load test against the running Compose stack: writes results/e3_load.json."""

import argparse
from pathlib import Path

from driftguard.benchmark.loadtest import run_e3
from driftguard.common.config import load_config
from driftguard.common.logging import configure_logging
from driftguard.common.results import write_results
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "e3.yaml"
NOTES = (
    "Docker Compose on one host (see hardware), 6 partitions per flow topic, one consumer "
    "instance. Producer runs on the host at a fixed rate. Retraining condition submits a real "
    "Prefect retraining run in the middle of each step. Stream features off replaces behavioral "
    "features with zeros to isolate their cost. Rollout mode, challenger alias, idempotency "
    "keys, and window state are reset before every step. Round-trip latency is measured by the "
    "consumer around each scoring request; model latency is the serving router's inference time "
    "only; both are quantiles over the whole step."
)


def main() -> None:
    """Run E3 and write its results file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    configure_logging("loadtest")
    metrics = run_e3(load_config(args.config).data)
    print(f"Wrote {write_results('e3_load', metrics, args.config, notes=NOTES)}")


if __name__ == "__main__":
    main()
