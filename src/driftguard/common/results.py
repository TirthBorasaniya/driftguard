"""Results artifact writer following CONVENTIONS 7 (one JSON file per run under results/)."""

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from driftguard.common.config import file_sha256
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

SCHEMA_VERSION = 1
TRACKED_PACKAGES = (
    "lightgbm",
    "numpy",
    "pandas",
    "scikit-learn",
    "onnxruntime",
    "onnxmltools",
    "mlflow",
    "prefect",
    "feast",
    "great-expectations",
    "evidently",
    "river",
    "confluent-kafka",
    "redis",
    "fastapi",
)


# ============= Environment capture =============


def git_sha(repo_root: Path = REPO_ROOT) -> str:
    """Return the short git SHA of HEAD, or `unknown` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def git_dirty(repo_root: Path = REPO_ROOT) -> bool:
    """Return True if tracked files differ from HEAD."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(out.stdout.strip())


def _sysctl(name: str) -> str | None:
    """Read a macOS sysctl value, or None when unavailable."""
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def describe_hardware() -> str:
    """
    Describe the host CPU and memory.

    Returns
    -------
    hardware : str
        Human-readable description, overridable with `DRIFTGUARD_HARDWARE`.
    """
    override = os.environ.get("DRIFTGUARD_HARDWARE")
    if override:
        return override
    cpu = _sysctl("machdep.cpu.brand_string") or platform.processor() or platform.machine()
    mem_gb = None
    mem_bytes = _sysctl("hw.memsize")
    if mem_bytes and mem_bytes.isdigit():
        mem_gb = int(mem_bytes) / 1024**3
    elif Path("/proc/meminfo").exists():
        first = Path("/proc/meminfo").read_text().splitlines()[0]
        mem_gb = int(first.split()[1]) / 1024**2
    cores = os.cpu_count()
    mem_text = f", {mem_gb:.0f} GB" if mem_gb else ""
    return f"{cpu}, {cores} logical CPUs{mem_text}"


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Return installed versions of the tracked packages that are present."""
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


# ============= Writer =============


def build_results(
    name: str,
    metrics: dict[str, Any],
    config_path: str | Path | None,
    notes: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Assemble a results document.

    Parameters
    ----------
    name : str
        Results name, also the file stem (for example `e1_model_quality`).
    metrics : dict
        Experiment metrics; structure is experiment specific.
    config_path : str, Path, or None
        Config that produced the run; its SHA-256 is recorded.
    notes : str
        Free-text notes (protocol caveats, provenance).
    now : datetime or None
        Timestamp override for tests.

    Returns
    -------
    document : dict
        Results document matching CONVENTIONS 7.
    """
    now = now or datetime.now(UTC)
    config_rel, config_hash = None, None
    if config_path is not None:
        config_path = Path(config_path)
        config_hash = file_sha256(config_path)
        try:
            config_rel = str(config_path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            config_rel = str(config_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"{now.strftime('%Y-%m-%dT%H-%M-%SZ')}_{name}",
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "timestamp_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_path": config_rel,
        "config_sha256": config_hash,
        "environment": {
            "python": platform.python_version(),
            "platform": sys.platform,
            "packages": package_versions(),
            "hardware": describe_hardware(),
        },
        "metrics": metrics,
        "notes": notes,
    }


def write_results(
    name: str,
    metrics: dict[str, Any],
    config_path: str | Path | None,
    notes: str = "",
    results_dir: str | Path = REPO_ROOT / "results",
) -> Path:
    """
    Write a results document to `results/<name>.json`.

    Parameters
    ----------
    name : str
        Results name and file stem.
    metrics : dict
        Experiment metrics.
    config_path : str, Path, or None
        Config that produced the run.
    notes : str
        Free-text notes.
    results_dir : str or Path
        Output directory.

    Returns
    -------
    path : Path
        Written file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    document = build_results(name, metrics, config_path, notes)
    path = results_dir / f"{name}.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=False, default=_json_default) + "\n")
    return path


def _json_default(value: Any) -> Any:
    """Serialize NumPy scalars and arrays."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def load_results(name: str, results_dir: str | Path = REPO_ROOT / "results") -> dict | None:
    """Load `results/<name>.json`, or None if it has not been produced."""
    path = Path(results_dir) / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
