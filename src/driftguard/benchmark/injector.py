"""Benchmark-only drift injector: covariate, prior, and concept shifts at a known onset."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ============= Constants =============

SHIFT_TYPES = ("covariate", "prior", "concept")


# ============= Manifest =============


@dataclass
class InjectionManifest:
    """Exact description of one injected shift, so detection delay is computed exactly."""

    shift_type: str
    onset: int
    params: dict[str, Any]
    seed: int
    n_rows: int
    rows_modified: int
    extra: dict[str, Any] = field(default_factory=dict)

    def write(self, path: Path) -> None:
        """Persist the manifest as JSON."""
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


# ============= Shifts =============


def covariate_shift(
    segment: pd.DataFrame,
    onset: int,
    features: list[str],
    factor: float,
    noise_std: float,
    reference_std: dict[str, float],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    """
    Multiply selected features by `factor` and add Gaussian noise from row `onset` onwards.

    Noise has standard deviation `noise_std * reference_std[feature]`, so it is scaled to each
    feature. Labels are unchanged.
    """
    out = segment.copy()
    n_after = len(out) - onset
    for name in features:
        values = out[name].to_numpy(dtype=np.float64)
        noise = rng.normal(0.0, noise_std * reference_std[name], size=n_after)
        values[onset:] = values[onset:] * factor + noise
        out[name] = values.astype(out[name].dtype)
    return out, n_after


def prior_shift(
    segment: pd.DataFrame,
    onset: int,
    attack_rate_after: float,
    attack_pool: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    """
    Resample rows after `onset` so the attack proportion equals `attack_rate_after`.

    Post-onset rows are redrawn (with replacement) from the segment's own benign rows and from
    `attack_pool`; the row count and event ids order stay stable so windows align.
    """
    before, after = segment.iloc[:onset], segment.iloc[onset:]
    n_after = len(after)
    n_attack = int(round(attack_rate_after * n_after))
    benign = after[after["label"] == 0]
    if benign.empty or attack_pool.empty:
        raise ValueError("prior shift needs benign rows after onset and a non-empty attack pool")
    drawn = pd.concat(
        [
            benign.sample(n_after - n_attack, replace=True, random_state=rng.integers(1 << 31)),
            attack_pool.sample(n_attack, replace=True, random_state=rng.integers(1 << 31)),
        ]
    )
    drawn = drawn.sample(frac=1.0, random_state=rng.integers(1 << 31))
    drawn = drawn.assign(event_id=[f"{eid}#p{i}" for i, eid in enumerate(drawn["event_id"])])
    return pd.concat([before, drawn], ignore_index=True), n_after


def concept_shift(
    segment: pd.DataFrame,
    onset: int,
    region_rule: dict[str, Any],
    flip_fraction: float,
    reference: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    """
    Relabel part of a feature-space region as attacks after `onset` (attacks that look benign).

    The region is `feature <= quantile(reference[feature], q)` for `region_rule = {"feature",
    "quantile"}`. A `flip_fraction` share of benign rows inside it becomes label 1 with family
    `injected_concept`. Features do not change, so only performance-based detectors can see it.
    """
    threshold = float(np.quantile(reference[region_rule["feature"]], region_rule["quantile"]))
    out = segment.copy()
    in_region = (out[region_rule["feature"]] <= threshold) & (out["label"] == 0)
    in_region.iloc[:onset] = False
    candidates = np.flatnonzero(in_region.to_numpy())
    flip = rng.choice(candidates, size=int(round(flip_fraction * len(candidates))), replace=False)
    out.loc[out.index[flip], "label"] = 1
    out.loc[out.index[flip], "label_family"] = "injected_concept"
    return out, int(len(flip))


def inject(
    segment: pd.DataFrame,
    shift_type: str,
    onset: int,
    params: dict[str, Any],
    seed: int,
    reference: pd.DataFrame,
    attack_pool: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, InjectionManifest]:
    """
    Apply one shift to a stationary segment.

    Parameters
    ----------
    segment : pd.DataFrame
        Stationary rows in stream order (features, `label`, `label_family`, `event_id`).
    shift_type : str
        `covariate`, `prior`, or `concept`.
    onset : int
        Row index where the shift starts.
    params : dict
        Shift parameters (SPEC 3.5 table).
    seed : int
        Random seed.
    reference : pd.DataFrame
        Reference rows used for feature scales and region quantiles.
    attack_pool : pd.DataFrame or None
        Attack rows for prior shifts.

    Returns
    -------
    shifted : pd.DataFrame
    manifest : InjectionManifest
    """
    if shift_type not in SHIFT_TYPES:
        raise ValueError(f"shift_type must be one of {SHIFT_TYPES}")
    if not 0 < onset < len(segment):
        raise ValueError("onset must fall inside the segment")
    rng = np.random.default_rng(seed)
    if shift_type == "covariate":
        std = {name: float(reference[name].std()) or 1.0 for name in params["features"]}
        shifted, modified = covariate_shift(
            segment,
            onset,
            params["features"],
            float(params["factor"]),
            float(params["noise_std"]),
            std,
            rng,
        )
    elif shift_type == "prior":
        shifted, modified = prior_shift(
            segment,
            onset,
            float(params["attack_rate_after"]),
            attack_pool if attack_pool is not None else pd.DataFrame(),
            rng,
        )
    else:
        shifted, modified = concept_shift(
            segment, onset, params["region_rule"], float(params["flip_fraction"]), reference, rng
        )
    manifest = InjectionManifest(
        shift_type, int(onset), dict(params), int(seed), len(shifted), int(modified)
    )
    return shifted, manifest
