"""Alias moves for rollback, promotion, and decline, each logged to promotions.jsonl (SPEC 7)."""

from pathlib import Path
from typing import Any

from driftguard.common.metrics import PROMOTIONS
from driftguard.common.mlflow_client import CHALLENGER, CHAMPION, PREVIOUS, Registry, model_name
from driftguard.common.promotions import append_promotion
from driftguard.serving.router import FRACTION_KEY, MODE_KEY, MODES

# ============= Errors =============


class RolloutConflictError(RuntimeError):
    """The requested alias move is not possible in the current state."""


# ============= Mode =============


def set_mode(redis_client, dataset: str, mode: str, canary_fraction: float | None = None) -> None:
    """Set the rollout mode (and optionally the canary fraction) in Redis."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    redis_client.set(MODE_KEY.format(dataset=dataset), mode)
    if canary_fraction is not None:
        if not 0.0 < canary_fraction < 1.0:
            raise ValueError("canary_fraction must be in (0, 1)")
        redis_client.set(FRACTION_KEY.format(dataset=dataset), str(canary_fraction))


# ============= Alias moves =============


def rollback(
    registry: Registry,
    redis_client,
    dataset: str,
    log_path: Path,
    reason: str = "operator rollback",
) -> dict[str, Any]:
    """
    Move `previous_champion` back to `champion` and return to `champion_only`.

    The rolled-back version keeps no alias; `previous_champion` is cleared because it now
    serves. The serving watcher swaps sessions within one poll interval.

    Raises
    ------
    RolloutConflictError
        If no previous champion exists.
    """
    name = model_name(dataset)
    previous = registry.get_alias(name, PREVIOUS)
    if previous is None:
        raise RolloutConflictError("no previous_champion to roll back to")
    current = registry.get_alias(name, CHAMPION)
    registry.set_alias(name, CHAMPION, previous.version)
    registry.delete_alias(name, PREVIOUS)
    registry.delete_alias(name, CHALLENGER)
    set_mode(redis_client, dataset, "champion_only")
    PROMOTIONS.labels(dataset=dataset, result="rolled_back").inc()
    return append_promotion(
        log_path,
        {
            "dataset": dataset,
            "action": "rollback",
            "champion": previous.version,
            "rolled_back_from": current.version if current else None,
            "reason": reason,
        },
    )


def promote(
    registry: Registry, redis_client, dataset: str, log_path: Path, evidence: dict[str, Any]
) -> dict[str, Any]:
    """
    After a passed canary: `champion` -> `previous_champion`, then `challenger` -> `champion`.

    Raises
    ------
    RolloutConflictError
        If there is no challenger.
    """
    name = model_name(dataset)
    challenger = registry.get_alias(name, CHALLENGER)
    if challenger is None:
        raise RolloutConflictError("no challenger to promote")
    champion = registry.get_alias(name, CHAMPION)
    if champion is not None:
        registry.set_alias(name, PREVIOUS, champion.version)
    registry.set_alias(name, CHAMPION, challenger.version)
    registry.delete_alias(name, CHALLENGER)
    set_mode(redis_client, dataset, "champion_only")
    PROMOTIONS.labels(dataset=dataset, result="promoted").inc()
    return append_promotion(
        log_path,
        {
            "dataset": dataset,
            "action": "promote",
            "champion": challenger.version,
            "previous_champion": champion.version if champion else None,
            **evidence,
        },
    )


def decline(
    registry: Registry,
    redis_client,
    dataset: str,
    log_path: Path,
    action: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove the challenger and return to `champion_only` (gate declined or canary rolled back).

    Parameters
    ----------
    action : str
        `decline` (shadow gate) or `canary_rolled_back` (guardrail breach).
    """
    name = model_name(dataset)
    challenger = registry.get_alias(name, CHALLENGER)
    champion = registry.get_alias(name, CHAMPION)
    registry.delete_alias(name, CHALLENGER)
    set_mode(redis_client, dataset, "champion_only")
    result = "declined" if action == "decline" else "rolled_back"
    PROMOTIONS.labels(dataset=dataset, result=result).inc()
    return append_promotion(
        log_path,
        {
            "dataset": dataset,
            "action": action,
            "challenger": challenger.version if challenger else None,
            "champion": champion.version if champion else None,
            **evidence,
        },
    )
