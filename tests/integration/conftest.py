"""Integration fixtures: the Compose stack must already be running and bootstrapped."""

import httpx
import pytest

from driftguard.common.settings import get_settings


@pytest.fixture(scope="session", autouse=True)
def stack_ready():
    """Fail fast with a clear message when the stack is not running."""
    settings = get_settings()
    for url in (f"{settings.serving_url}/readyz", "http://localhost:9101/healthz"):
        try:
            httpx.get(url, timeout=5).raise_for_status()
        except httpx.HTTPError as exc:
            pytest.exit(f"stack not ready ({url}): {exc}; run `make up && make bootstrap`")
