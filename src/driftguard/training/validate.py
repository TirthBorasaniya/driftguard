"""Great Expectations suite with bounds fitted once on the initial reference data (SPEC 6.2)."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ============= Constants =============

LOWER_PERCENTILE = 0.1
UPPER_PERCENTILE = 99.9


FLOAT32_MAX = 3.4e38


# ============= Bounds =============


def physical_bounds(dataset: str) -> dict[str, list[float]]:
    """
    Physical validity limits per feature from the dataset config (never fitted to data).

    A feature without a configured minimum or maximum gets the float32 range, so the suite still
    rejects infinities and nulls. Unlike percentile bounds, these limits describe broken data
    (negative durations, ports above 65535, non-finite values), not drifted data: a window whose
    distribution has moved still passes.

    Parameters
    ----------
    dataset : str
        Dataset name.

    Returns
    -------
    bounds : dict
        Feature name to `[low, high]`.
    """
    from driftguard.data.registry import feature_specs

    return {
        spec["name"]: [
            -FLOAT32_MAX if spec.get("min") is None else float(spec["min"]),
            FLOAT32_MAX if spec.get("max") is None else float(spec["max"]),
        ]
        for spec in feature_specs(dataset)
    }


def fit_bounds(
    reference: pd.DataFrame,
    features: list[str],
    lower: float = LOWER_PERCENTILE,
    upper: float = UPPER_PERCENTILE,
) -> dict[str, list[float]]:
    """
    Fit per-feature value bounds as the 0.1 and 99.9 percentiles of the reference data.

    Returns
    -------
    bounds : dict
        Maps feature name to `[low, high]`.
    """
    values = reference[features].to_numpy(dtype=np.float64)
    lows = np.percentile(values, lower, axis=0)
    highs = np.percentile(values, upper, axis=0)
    return {
        name: [float(lo), float(hi)] for name, lo, hi in zip(features, lows, highs, strict=True)
    }


def save_bounds(path: Path, bounds: dict[str, list[float]], meta: dict) -> None:
    """Write the bounds file; it is fitted once and never loosened afterwards."""
    path.write_text(json.dumps({"meta": meta, "bounds": bounds}, indent=2) + "\n")


def load_bounds(path: Path) -> dict[str, list[float]]:
    """Load fitted bounds; raises a clear error when `scripts/fit_gx_bounds.py` was not run."""
    if not path.exists():
        raise FileNotFoundError(f"GX bounds not found at {path}; run `make fit-gx-bounds`")
    return json.loads(path.read_text())["bounds"]


# ============= Suite =============


@dataclass
class ValidationReport:
    """Result of running the suite on one frame."""

    success: bool
    failed: list[str]
    evaluated: int


class GXValidator:
    """
    Runs a Great Expectations 1.x suite on pandas frames, reusing one ephemeral context.

    Expectations: the feature columns exist, values are not null, and each feature lies within
    its fitted bounds for at least `mostly` of the rows; the row count is at least `min_rows`.

    Parameters
    ----------
    bounds : dict
        Feature name to `[low, high]`.
    mostly : float
        Required in-bounds share per feature.
    min_rows : int
        Minimum row count.
    name : str
        Suite name.
    """

    def __init__(
        self,
        bounds: dict[str, list[float]],
        mostly: float,
        min_rows: int,
        name: str = "driftguard_window",
    ):
        import great_expectations as gx
        from great_expectations import expectations as gxe

        self.features = list(bounds)
        self.context = gx.get_context(mode="ephemeral")
        source = self.context.data_sources.add_pandas(f"{name}_source")
        asset = source.add_dataframe_asset(f"{name}_asset")
        self.batch_definition = asset.add_batch_definition_whole_dataframe(f"{name}_batch")
        suite = gx.ExpectationSuite(name=name)
        suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=min_rows))
        for feature, (low, high) in bounds.items():
            suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=feature))
            suite.add_expectation(
                gxe.ExpectColumnValuesToBeBetween(
                    column=feature, min_value=low, max_value=high, mostly=mostly
                )
            )
        self.suite = self.context.suites.add(suite)

    def validate(self, frame: pd.DataFrame) -> ValidationReport:
        """Validate a frame containing at least the bounded feature columns."""
        missing = [name for name in self.features if name not in frame.columns]
        if missing:
            return ValidationReport(False, [f"missing_column:{name}" for name in missing], 0)
        batch = self.batch_definition.get_batch(
            batch_parameters={"dataframe": frame[self.features]}
        )
        result = batch.validate(self.suite)
        failed = []
        for item in result.results:
            if not item.success:
                config = item.expectation_config
                column = config.kwargs.get("column", "table")
                failed.append(f"{config.type}:{column}")
        return ValidationReport(bool(result.success), failed, len(result.results))
