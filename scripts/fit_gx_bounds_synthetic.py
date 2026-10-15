"""CI only: write GX bounds from synthetic flows when no bounds file exists (never committed)."""

import argparse
import sys

from driftguard.common.settings import REPO_ROOT
from driftguard.data.registry import feature_names
from driftguard.data.synthetic import synthetic_flows
from driftguard.training.validate import fit_bounds, save_bounds

# ============= CLI =============


def main() -> int:
    """Write synthetic bounds for CI stacks that run without the real dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cicids2017")
    parser.add_argument("--if-missing", action="store_true", dest="o_if_missing")
    args = parser.parse_args()
    path = REPO_ROOT / "configs" / "data" / f"{args.dataset}_gx_bounds.json"
    if path.exists():
        print(f"{path} exists; leaving it unchanged")
        return 0
    frame = synthetic_flows(args.dataset, 20_000, attack_rate=0.1, seed=7)
    save_bounds(
        path,
        fit_bounds(frame, feature_names(args.dataset)),
        {"dataset": args.dataset, "source": "synthetic (CI only)"},
    )
    print(f"Wrote CI-only bounds to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
