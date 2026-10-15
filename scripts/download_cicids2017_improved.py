"""Download the DistriNet improved CICIDS2017 archive, verify its SHA-256, and extract it."""

import argparse
import sys
import zipfile

from driftguard.common.config import file_sha256, load_config
from driftguard.common.download import download
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

CONFIG = REPO_ROOT / "configs" / "data" / "cicids2017.yaml"
CHUNK = 1 << 20


# ============= Download =============


def main() -> int:
    """Download, verify, and extract; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o-force", action="store_true", dest="o_force")
    args = parser.parse_args()

    source = load_config(CONFIG)["sources"]["improved"]
    raw_dir = REPO_ROOT / source["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / source["archive"]
    if args.o_force or not archive.exists():
        print(f"Downloading {source['url']}")
        download(source["url"], archive)
    digest = file_sha256(archive)
    if digest != source["sha256"]:
        print(f"SHA-256 mismatch: expected {source['sha256']}, got {digest}")
        return 1
    print(f"SHA-256 verified: {digest}")
    with zipfile.ZipFile(archive) as handle:
        for member in source["files"].values():
            handle.extract(member, raw_dir)
    print(f"Extracted {len(source['files'])} files to {raw_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
