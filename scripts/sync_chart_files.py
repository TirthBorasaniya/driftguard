"""Copy alert rules and dashboards into the Helm chart's files/ (`--check` verifies the copies)."""

import argparse
import shutil
import sys

from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

CHART_FILES = REPO_ROOT / "deploy" / "k8s" / "charts" / "driftguard" / "files"
SOURCES = {REPO_ROOT / "prometheus" / "alerts.yml": CHART_FILES / "alerts.yml"}
SOURCES.update(
    {
        path: CHART_FILES / "dashboards" / path.name
        for path in sorted((REPO_ROOT / "grafana" / "dashboards").glob("*.json"))
    }
)


# ============= CLI =============


def main() -> int:
    """Copy or check; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", dest="o_check")
    args = parser.parse_args()
    stale = [
        dst
        for src, dst in SOURCES.items()
        if not dst.exists() or dst.read_bytes() != src.read_bytes()
    ]
    if args.o_check:
        if stale:
            print(f"Stale chart files: {[str(p.relative_to(REPO_ROOT)) for p in stale]}")
            return 1
        print("Chart files up to date")
        return 0
    for src, dst in SOURCES.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print(f"Synced {len(SOURCES)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
