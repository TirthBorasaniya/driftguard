"""Rollout routing: champion only, shadow, and deterministic canary split (SPEC 4.2)."""

import hashlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from driftguard.common.metrics import (
    POSITIVE_DECISIONS,
    ROLLOUT_MODE,
    SCORED_EVENTS,
    SCORING_ERRORS,
    SCORING_LATENCY,
    SCORING_REQUESTS,
)
from driftguard.common.mlflow_client import CHALLENGER, CHAMPION
from driftguard.serving.model_store import LoadedModel, ModelStore

# ============= Constants =============

MODES = ("champion_only", "shadow", "canary")
MODE_KEY = "dg:rollout:{dataset}"
FRACTION_KEY = "dg:rollout:{dataset}:canary_fraction"
HASH_SPACE = float(1 << 32)


# ============= Routing =============


def canary_bucket(event_id: str) -> float:
    """Deterministic position of an event in [0, 1) from SHA-256 of its id (never `hash()`)."""
    return int(hashlib.sha256(event_id.encode()).hexdigest()[:8], 16) / HASH_SPACE


def routes_to_canary(event_id: str, fraction: float) -> bool:
    """True when the event falls into the canary share."""
    return canary_bucket(event_id) < fraction


class NoChampionError(RuntimeError):
    """No champion is loaded yet."""


@dataclass
class RolloutState:
    """Current rollout mode and canary fraction, read from Redis."""

    mode: str
    canary_fraction: float


class Router:
    """
    Scores requests according to the rollout mode.

    Parameters
    ----------
    store : ModelStore
        Served models.
    redis_client : redis.Redis
        Holds `dg:rollout:{dataset}` and the canary fraction.
    dataset : str
        Dataset served.
    default_fraction : float
        Canary fraction when Redis has none.
    recent_size : int
        Size of the recent-predictions buffer (ids, scores, and versions only).
    """

    def __init__(
        self,
        store: ModelStore,
        redis_client,
        dataset: str,
        default_fraction: float = 0.10,
        recent_size: int = 1000,
        state_ttl_s: float = 1.0,
    ):
        self.store = store
        self.redis = redis_client
        self.dataset = dataset
        self.default_fraction = default_fraction
        self.recent: deque = deque(maxlen=recent_size)
        self._state_ttl_s = state_ttl_s
        self._state: RolloutState | None = None
        self._state_at = 0.0

    def state(self) -> RolloutState:
        """Read the rollout mode with a short cache so Redis is not hit on every request."""
        now = time.monotonic()
        if self._state is None or now - self._state_at > self._state_ttl_s:
            mode = self.redis.get(MODE_KEY.format(dataset=self.dataset)) or "champion_only"
            fraction = self.redis.get(FRACTION_KEY.format(dataset=self.dataset))
            mode = mode if mode in MODES else "champion_only"
            self._state = RolloutState(mode, float(fraction) if fraction else self.default_fraction)
            self._state_at = now
            for name in MODES:
                ROLLOUT_MODE.labels(dataset=self.dataset, mode=name).set(int(name == mode))
        return self._state

    def invalidate(self) -> None:
        """Drop the cached state after an admin change."""
        self._state = None

    def _score(self, model: LoadedModel, names: tuple[str, ...], matrix: np.ndarray) -> np.ndarray:
        SCORING_REQUESTS.labels(dataset=self.dataset, alias=model.alias).inc()
        started = time.perf_counter()
        try:
            scores = model.score(names, matrix)
        except Exception:
            SCORING_ERRORS.labels(dataset=self.dataset, alias=model.alias).inc()
            raise
        SCORING_LATENCY.labels(dataset=self.dataset, alias=model.alias).observe(
            time.perf_counter() - started
        )
        decisions = scores >= model.threshold
        SCORED_EVENTS.labels(dataset=self.dataset, alias=model.alias).inc(len(scores))
        POSITIVE_DECISIONS.labels(dataset=self.dataset, alias=model.alias).inc(int(decisions.sum()))
        return scores

    def score(self, event_ids: list[str], names: list[str], matrix: np.ndarray) -> dict[str, Any]:
        """
        Score a batch with the champion, or split it between champion and challenger in canary.

        Returns
        -------
        response : dict
            `rollout_mode`, `results`, and `latency_ms` (inference time of the answering models).
        """
        started = time.perf_counter()
        state = self.state()
        champion = self.store.get(CHAMPION)
        if champion is None:
            raise NoChampionError("champion not loaded")
        challenger = self.store.get(CHALLENGER) if state.mode == "canary" else None
        key = tuple(names)
        assigned = [champion] * len(event_ids)
        if challenger is not None:
            canary_mask = np.array(
                [routes_to_canary(eid, state.canary_fraction) for eid in event_ids], dtype=bool
            )
        else:
            canary_mask = np.zeros(len(event_ids), dtype=bool)
        scores = np.empty(len(event_ids), dtype=np.float64)
        if (~canary_mask).any():
            scores[~canary_mask] = self._score(champion, key, matrix[~canary_mask])
        if canary_mask.any():
            scores[canary_mask] = self._score(challenger, key, matrix[canary_mask])
            assigned = [challenger if m else champion for m in canary_mask]
        results = []
        for eid, score, model in zip(event_ids, scores, assigned, strict=True):
            item = {
                "event_id": eid,
                "score": float(score),
                "decision": int(score >= model.threshold),
                "threshold": model.threshold,
                "model_alias": model.alias,
                "model_version": model.version,
            }
            results.append(item)
            self.recent.append(
                {
                    k: item[k]
                    for k in ("event_id", "score", "decision", "model_alias", "model_version")
                }
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {"rollout_mode": state.mode, "results": results, "latency_ms": latency_ms}

    def score_shadow(
        self, event_ids: list[str], names: list[str], matrix: np.ndarray
    ) -> dict[str, Any] | None:
        """Score with the challenger while in shadow mode; None when shadow is not active."""
        challenger = self.store.get(CHALLENGER)
        if self.state().mode != "shadow" or challenger is None:
            return None
        scores = self._score(challenger, tuple(names), matrix)
        return {
            "model_version": challenger.version,
            "results": [
                {"event_id": e, "score": float(s)} for e, s in zip(event_ids, scores, strict=True)
            ],
        }
