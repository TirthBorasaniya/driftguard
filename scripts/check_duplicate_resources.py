"""Fail if a rendered manifest (file or stdin) declares the same Kubernetes resource twice."""

import argparse
import sys
from pathlib import Path

from driftguard.common.manifests import find_duplicate_resources

# ============= CLI =============


def main() -> int:
    """Print every duplicated resource; returns 1 if there is any, otherwise 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, help="rendered YAML; default stdin")
    args = parser.parse_args()
    rendered = args.manifest.read_text() if args.manifest else sys.stdin.read()
    duplicates = find_duplicate_resources(rendered)
    for kind, namespace, name, count in duplicates:
        print(f"duplicate {kind} {namespace}/{name} rendered {count} times")
    if duplicates:
        return 1
    print("No duplicate resources rendered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
