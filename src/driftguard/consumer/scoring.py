"""Serving router client with timeouts, jittered retries, and shadow calls (SPEC 4.1)."""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from driftguard.common.logging import get_logger

# ============= Constants =============

logger = get_logger("consumer.scoring")
SHADOW_MODE = "shadow"


# ============= Errors and results =============


class ScoringUnavailableError(Exception):
    """Scoring failed after all retries."""


@dataclass
class ScoreResponse:
    """Champion (or canary) scores for one request."""

    rollout_mode: str
    results: list[dict[str, Any]]
    latency_ms: float


# ============= Client =============


def backoff_delay(attempt: int, base_s: float, max_s: float, rng: random.Random) -> float:
    """Full-jitter exponential backoff: uniform in [0, min(max_s, base_s * 2**attempt)]."""
    return rng.uniform(0.0, min(max_s, base_s * (2**attempt)))


class ScoringClient:
    """
    Calls `POST /score` and `POST /score/shadow` on the serving router.

    Parameters
    ----------
    base_url : str
        Serving router URL.
    dataset : str
        Dataset name sent with each request.
    timeout_s : float
        Per-request timeout (SPEC: 2 s).
    retries : int
        Retries after the first attempt (SPEC: 3).
    backoff_base_s, backoff_max_s : float
        Jittered backoff parameters.
    shadow_timeout_s : float
        Timeout for shadow requests; shadow failures never fail the batch.
    client : httpx.Client or None
        Injectable HTTP client.
    sleep : callable
        Injectable sleep for tests.
    """

    def __init__(
        self,
        base_url: str,
        dataset: str,
        timeout_s: float = 2.0,
        retries: int = 3,
        backoff_base_s: float = 0.2,
        backoff_max_s: float = 2.0,
        shadow_timeout_s: float = 2.0,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        seed: int | None = None,
    ):
        self.dataset = dataset
        self.retries = int(retries)
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.timeout_s = timeout_s
        self.shadow_timeout_s = shadow_timeout_s
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self._sleep = sleep
        self._rng = random.Random(seed)
        self.last_mode = "champion_only"

    @staticmethod
    def payload(
        dataset: str, events: list[tuple[str, list[float]]], feature_names: list[str]
    ) -> dict[str, Any]:
        """
        Build the request body: feature names once, then event ids with ordered vectors.

        The serving model selects its own columns by name, so a model trained with or without
        stream features accepts the same request. No IPs or other payload fields are sent.
        """
        return {
            "dataset": dataset,
            "feature_names": feature_names,
            "events": [{"event_id": eid, "features": feats} for eid, feats in events],
        }

    def score(
        self, events: list[tuple[str, list[float]]], feature_names: list[str]
    ) -> ScoreResponse:
        """
        Score up to 256 events, retrying with jittered backoff.

        Raises
        ------
        ScoringUnavailableError
            After `1 + retries` failed attempts.

        Parameters
        ----------
        events : list[tuple[str, list[float]]]
            (event id, feature vector) pairs.
        feature_names : list[str]
            Names of the vector columns, in order.
        """
        body = self.payload(self.dataset, events, feature_names)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._client.post("/score", json=body, timeout=self.timeout_s)
                response.raise_for_status()
                data = response.json()
                if len(data["results"]) != len(events):
                    raise ValueError("result count does not match request")
                self.last_mode = data.get("rollout_mode", "champion_only")
                return ScoreResponse(self.last_mode, data["results"], float(data["latency_ms"]))
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                last_error = exc
                logger.warning("scoring_attempt_failed", attempt=attempt, error=str(exc))
                if attempt < self.retries:
                    self._sleep(
                        backoff_delay(attempt, self.backoff_base_s, self.backoff_max_s, self._rng)
                    )
        raise ScoringUnavailableError(str(last_error))

    def score_shadow(
        self, events: list[tuple[str, list[float]]], feature_names: list[str]
    ) -> dict[str, Any] | None:
        """
        Score the same events with the challenger; returns None when unavailable.

        Returns
        -------
        shadow : dict or None
            `{"model_version": str, "scores": {event_id: score}}`.
        """
        body = self.payload(self.dataset, events, feature_names)
        try:
            response = self._client.post("/score/shadow", json=body, timeout=self.shadow_timeout_s)
            if response.status_code == 409:
                return None
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("shadow_scoring_failed", error=str(exc))
            raise
        return {
            "model_version": str(data["model_version"]),
            "scores": {item["event_id"]: float(item["score"]) for item in data["results"]},
        }
