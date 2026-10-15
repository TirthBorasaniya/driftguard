"""Generate the committed Avro schemas under src/driftguard/data/schemas from the configs."""

import argparse
import sys

from driftguard.data.registry import DATASETS, dataset_config_path
from driftguard.data.schemas import SCHEMA_DIR, all_schemas, render_schema, schema_path

# ============= CLI =============


def main() -> int:
    """Write or check schema files; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", dest="o_check")
    args = parser.parse_args()

    datasets = tuple(ds for ds in DATASETS if dataset_config_path(ds).exists())
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for subject, schema in all_schemas(datasets).items():
        path, text = schema_path(subject), render_schema(schema)
        if path.exists() and path.read_text() == text:
            continue
        stale.append(subject)
        if not args.o_check:
            path.write_text(text)
    if args.o_check and stale:
        print(f"Stale schemas: {stale}")
        return 1
    print(f"Schemas up to date ({len(datasets)} datasets)" if not stale else f"Wrote {stale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
