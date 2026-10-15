"""Shared pytest fixtures."""

import os
from pathlib import Path

import pytest

# keep matplotlib and other caches inside the repository
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "mpl"))

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    """Repository root path."""
    return REPO_ROOT
