"""Deterministic emission plan: flows, delayed labels, and injected duplicates (SPEC 3.4)."""

import heapq
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

# ============= Constants =============

FLOW, LABEL, DUPLICATE = "flow", "label", "duplicate"
# heap tie-break so that at equal release index labels precede duplicates
KIND_ORDER = {LABEL: 0, DUPLICATE: 1}


# ============= Plan =============


@dataclass(frozen=True)
class PlanConfig:
    """Parameters of the emission plan."""

    label_delay_min: int = 2000
    label_delay_max: int = 10000
    label_coverage: float = 1.0
    duplicate_rate: float = 0.0
    duplicate_delay_max: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 <= self.label_delay_min <= self.label_delay_max:
            raise ValueError("require 0 <= label_delay_min <= label_delay_max")
        if not 0.0 <= self.label_coverage <= 1.0:
            raise ValueError("label_coverage must be in [0, 1]")
        if not 0.0 <= self.duplicate_rate <= 1.0:
            raise ValueError("duplicate_rate must be in [0, 1]")


def draw_plan_arrays(n_events: int, config: PlanConfig) -> dict[str, np.ndarray]:
    """
    Draw every random choice of the plan up front from one seeded generator.

    Returns
    -------
    arrays : dict
        `labeled` (bool, shape (n,)), `label_delay` (int64, shape (n,)), `duplicated` (bool),
        `duplicate_delay` (int64). Delays are in events, inclusive bounds.
    """
    rng = np.random.default_rng(config.seed)
    return {
        "labeled": rng.random(n_events) < config.label_coverage,
        "label_delay": rng.integers(
            config.label_delay_min, config.label_delay_max + 1, n_events, dtype=np.int64
        ),
        "duplicated": rng.random(n_events) < config.duplicate_rate,
        "duplicate_delay": rng.integers(
            1, config.duplicate_delay_max + 1, n_events, dtype=np.int64
        ),
    }


def emission_plan(n_events: int, config: PlanConfig) -> Iterator[tuple[str, int]]:
    """
    Yield the ordered emission sequence for `n_events` flows.

    Each flow `i` is emitted at step `i`. Its label, if covered, is released after the flow at
    index `i + delay` has been emitted; an injected duplicate likewise after `i + d`. Items still
    pending after the last flow are flushed in release order.

    Parameters
    ----------
    n_events : int
        Number of flows in the replay.
    config : PlanConfig
        Plan parameters.

    Yields
    ------
    item : tuple of (str, int)
        (`flow` | `label` | `duplicate`, flow index).
    """
    arrays = draw_plan_arrays(n_events, config)
    pending: list[tuple[int, int, int, str]] = []
    for i in range(n_events):
        yield FLOW, i
        if arrays["labeled"][i]:
            heapq.heappush(
                pending, (i + int(arrays["label_delay"][i]), KIND_ORDER[LABEL], i, LABEL)
            )
        if arrays["duplicated"][i]:
            release = i + int(arrays["duplicate_delay"][i])
            heapq.heappush(pending, (release, KIND_ORDER[DUPLICATE], i, DUPLICATE))
        while pending and pending[0][0] <= i:
            _, _, index, kind = heapq.heappop(pending)
            yield kind, index
    while pending:
        _, _, index, kind = heapq.heappop(pending)
        yield kind, index
