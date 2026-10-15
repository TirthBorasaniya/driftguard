"""E3b Kubernetes autoscaling on kind: writes results/e3b_k8s_autoscaling.json and a figure."""

import argparse
from pathlib import Path

from driftguard.benchmark.figures import replica_timeline
from driftguard.benchmark.k8s_autoscaling import run_e3b
from driftguard.common.config import load_config
from driftguard.common.logging import configure_logging
from driftguard.common.results import write_results
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "e3b.yaml"
NOTES = (
    "kind on the documented host (single node), Strimzi KRaft with one broker, 6 partitions, "
    "KEDA ScaledObject (lagThreshold 5000, 1 to 6 replicas, cooldown 60 s), cooperative-sticky "
    "assignment. Duplicates and losses compare distinct produced event ids with distinct event "
    "ids in predictions.v1."
)


def main() -> None:
    """Run E3b and write its results file and replica chart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--o-reduced",
        action="store_true",
        dest="o_reduced",
        help="scale durations down for the nightly CI job",
    )
    args = parser.parse_args()
    configure_logging("loadtest-k8s")
    config = load_config(args.config).data
    metrics = run_e3b(config, args.o_reduced)
    figure = replica_timeline(
        metrics["samples"], REPO_ROOT / config["figures_dir"] / "e3b_replicas.png"
    )
    metrics["figure"] = str(figure.relative_to(REPO_ROOT))
    name = "e3b_k8s_autoscaling" + ("_reduced" if args.o_reduced else "")
    print(f"Wrote {write_results(name, metrics, args.config, notes=NOTES)}")


if __name__ == "__main__":
    main()
