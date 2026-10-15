"""Download the AnoShift Kyoto 2006+ archive, record or verify its SHA-256, extract the subsets."""

import argparse
import re
import sys
import tarfile
from pathlib import Path

from driftguard.common.config import file_sha256, load_config
from driftguard.common.download import download
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

CONFIG = REPO_ROOT / "configs" / "data" / "kyoto.yaml"
CHUNK = 1 << 20


# ============= Steps =============


def record_sha256(digest: str) -> None:
    """Write the digest into configs/data/kyoto.yaml (first download only)."""
    text = CONFIG.read_text()
    CONFIG.write_text(re.sub(r"^sha256: null$", f"sha256: {digest}", text, flags=re.MULTILINE))


def extract_subsets(archive: Path, raw_dir: Path, member_dir: str) -> list[str]:
    """Extract only the subset parquet files, refusing paths that escape the target."""
    extracted = []
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if not (
                member.isfile()
                and member.name.startswith(member_dir + "/")
                and member.name.endswith(".parquet")
            ):
                continue
            target = (raw_dir / member.name).resolve()
            if not str(target).startswith(str(raw_dir.resolve())):
                raise ValueError(f"unsafe member path {member.name}")
            handle.extract(member, raw_dir, filter="data")
            extracted.append(member.name)
    return extracted


def main() -> int:
    """Download, hash, extract; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--o-force", action="store_true", dest="o_force")
    args = parser.parse_args()
    config = load_config(CONFIG)
    raw_dir = REPO_ROOT / config["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / config["archive"]
    if args.o_force or not archive.exists():
        print(f"Downloading {config['url']} (about 8.1 GB)")
        download(config["url"], archive)
    digest = file_sha256(archive)
    if config["sha256"] is None:
        record_sha256(digest)
        print(f"Recorded SHA-256 {digest} in {CONFIG}")
    elif digest != config["sha256"]:
        print(f"SHA-256 mismatch: expected {config['sha256']}, got {digest}")
        return 1
    names = extract_subsets(archive, raw_dir, config["member_dir"])
    print(f"Extracted {len(names)} subset files to {raw_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
