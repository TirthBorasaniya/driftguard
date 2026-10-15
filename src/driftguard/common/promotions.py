"""Append-only log of alias changes (`results/promotions.jsonl`, SPEC 7.3)."""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ============= Constants =============

_LOCK = threading.Lock()


# ============= Log =============


def append_promotion(path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    """
    Append one alias-change record with a UTC timestamp.

    Parameters
    ----------
    path : str or Path
        Log file; created if missing, never rewritten.
    record : dict
        At least `dataset`, `action`, and the versions involved.

    Returns
    -------
    entry : dict
        The record as written.
    """
    entry = {"ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), **record}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def read_promotions(path: str | Path) -> list[dict[str, Any]]:
    """Read every record of the log, oldest first."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
