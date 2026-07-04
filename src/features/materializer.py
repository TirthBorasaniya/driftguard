"""Materialize offline Feast features to the Redis online store."""

import subprocess
import sys

from src.config import FEATURE_REPO_DIR


def apply_feast() -> bool:
    """
    Run feast apply to register feature definitions.

    Returns
    -------
    success : bool
    """
    result = subprocess.run(
        [sys.executable, "-m", "feast", "apply"],
        cwd=str(FEATURE_REPO_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"feast apply failed:\n{result.stderr}")
    print(result.stdout)
    return True


def materialize_to_online_store() -> bool:
    """
    Push all feature views to the Redis online store.

    Returns
    -------
    success : bool
    """
    from datetime import datetime

    end_date = datetime.now().isoformat()
    result = subprocess.run(
        [sys.executable, "-m", "feast", "materialize-incremental", end_date],
        cwd=str(FEATURE_REPO_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"feast materialize failed:\n{result.stderr}")
    print(result.stdout)
    return True


if __name__ == "__main__":
    apply_feast()
    materialize_to_online_store()
