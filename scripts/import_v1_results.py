"""Copy the first version's verified numbers from configs/eval/v1_import.yaml into results/."""

import argparse
from pathlib import Path

from driftguard.common.config import load_config
from driftguard.common.results import write_results
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval" / "v1_import.yaml"


# ============= CLI =============


def main() -> None:
    """Write results/v1_verified_run.json from the transcription record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = {
        "split": config["split"].strip(),
        "source_commit": config["source_commit"],
        "items": config["items"],
    }
    notes = (
        f"historical numbers from {config['source_document']} at commit "
        f"{config['source_commit']}; split: {metrics['split']}"
    )
    path = write_results("v1_verified_run", metrics, args.config, notes=notes)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
