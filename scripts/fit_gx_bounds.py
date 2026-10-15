"""Fit GX bounds once on the initial reference data (0.1 and 99.9 percentiles, SPEC 6.2)."""

import argparse
import sys

from driftguard.common.config import file_sha256
from driftguard.common.settings import REPO_ROOT
from driftguard.data.registry import dataset_config, dataset_config_path, feature_names
from driftguard.training.validate import LOWER_PERCENTILE, UPPER_PERCENTILE, fit_bounds, save_bounds

# ============= CLI =============


def reference_frame(dataset: str):
    """Load the initial reference data: the initial champion's training period."""
    if dataset == "cicids2017":
        from driftguard.data.cicids2017 import load_dataset

        frame, _ = load_dataset(days=list(dataset_config(dataset)["gx"]["reference_days"]))
        return frame
    from driftguard.data.kyoto import load_years

    return load_years(list(dataset_config(dataset)["gx"]["reference_years"]))


def main() -> int:
    """Write configs/data/<dataset>_gx_bounds.json; refuses to overwrite without --o-force."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cicids2017")
    parser.add_argument("--o-force", action="store_true", dest="o_force")
    args = parser.parse_args()
    path = REPO_ROOT / "configs" / "data" / f"{args.dataset}_gx_bounds.json"
    if path.exists() and not args.o_force:
        print(f"{path} exists; bounds are fitted once and never refitted to make a run pass")
        return 1
    frame = reference_frame(args.dataset)
    features = feature_names(args.dataset)
    meta = {
        "dataset": args.dataset,
        "percentiles": [LOWER_PERCENTILE, UPPER_PERCENTILE],
        "reference_rows": len(frame),
        "data_config_sha256": file_sha256(dataset_config_path(args.dataset)),
    }
    save_bounds(path, fit_bounds(frame, features), meta)
    print(f"Wrote {path} from {len(frame)} reference rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
