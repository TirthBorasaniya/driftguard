"""Rewrite README result tables from results/*.json; `--check` fails if the README is stale."""

import argparse
import sys
from pathlib import Path

from driftguard.common.render import render_readme
from driftguard.common.settings import REPO_ROOT

# ============= CLI =============


def main() -> int:
    """Render or check README tables; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--check", action="store_true", dest="o_check")
    args = parser.parse_args()

    current = args.readme.read_text()
    rendered = render_readme(current, args.results_dir)
    if args.o_check:
        if rendered != current:
            print("README result tables are stale; run `make results`.")
            return 1
        print("README result tables are up to date.")
        return 0
    if rendered != current:
        args.readme.write_text(rendered)
        print(f"Updated {args.readme}")
    else:
        print("README already up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
