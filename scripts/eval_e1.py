"""E1 model quality on CICIDS2017: writes results/e1_model_quality.json (SPEC 9, E1)."""

import argparse
from pathlib import Path

from driftguard.benchmark.model_quality import run_e1
from driftguard.common.config import load_config
from driftguard.common.logging import configure_logging
from driftguard.common.results import write_results
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "e1.yaml"
NOTES = (
    "Primary protocol is time-ordered (train Monday to Wednesday, test Thursday and Friday with "
    "unseen attack families). The random stratified split is an upper-bound reference only."
)


# ============= CLI =============


def main() -> None:
    """Run E1 and write its results file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    configure_logging("eval-e1")
    config = load_config(args.config)

    stream_feature_fn = None
    if config["ablations"].get("o_stream_features"):
        from driftguard.streamfeatures.offline import redis_replay_features

        stream_feature_fn = redis_replay_features
    metrics = run_e1(config, stream_feature_fn)
    print(f"Wrote {write_results('e1_model_quality', metrics, args.config, notes=NOTES)}")


if __name__ == "__main__":
    main()
