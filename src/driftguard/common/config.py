"""YAML configuration loading with version checks and content hashing."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ============= Loading =============


@dataclass(frozen=True)
class LoadedConfig:
    """A parsed YAML config together with its path and SHA-256 digest."""

    path: Path
    sha256: str
    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Return a top-level key or a default."""
        return self.data.get(key, default)


def file_sha256(path: str | Path) -> str:
    """
    Compute the SHA-256 digest of a file.

    Parameters
    ----------
    path : str or Path
        File to hash.

    Returns
    -------
    digest : str
        Hex-encoded SHA-256 digest.
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_config(path: str | Path) -> LoadedConfig:
    """
    Load a YAML config and verify it carries a `version` field.

    Parameters
    ----------
    path : str or Path
        Path to a YAML file under `configs/`.

    Returns
    -------
    config : LoadedConfig
        Parsed content, path, and SHA-256 of the file bytes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict) or "version" not in data:
        raise ValueError(f"config {path} must be a mapping with a 'version' field")
    return LoadedConfig(path=path, sha256=file_sha256(path), data=data)
