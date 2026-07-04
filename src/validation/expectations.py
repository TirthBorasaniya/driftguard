"""Great Expectations style validation suite: bounded data quality checks before retraining."""

import json
from pathlib import Path

import pandas as pd

from src.config import FEATURE_COLS

# ============= Suite Configuration =============

SUITE_NAME = "network_flow_feature_suite"

# the ten network flow feature columns subject to null-rate assertions
NUMERICAL_FEATURE_COLS = list(FEATURE_COLS)

MAX_NULL_RATE = 0.02  # at most 2% nulls per feature column
MIN_ROW_COUNT = 10_000
MAX_ROW_COUNT = 5_000_000

# per-feature value-range bounds; None means unbounded on that side
FEATURE_BOUNDS: dict[str, tuple[float, float | None]] = {
    "flow_duration": (0.0, None),
    "flow_bytes_per_sec": (0.0, None),
    "flow_packets_per_sec": (0.0, None),
    "total_fwd_packets": (0.0, None),
    "total_bwd_packets": (0.0, None),
    "packet_length_mean": (0.0, None),
    "packet_length_std": (0.0, None),
    "flow_iat_mean": (0.0, None),
    "fwd_bwd_packet_ratio": (0.0, None),
    "syn_flag_count": (0.0, None),
}


# ============= Bounded Assertions =============


def _check_bounds(
    df: pd.DataFrame,
    col: str,
    min_val: float | None,
    max_val: float | None,
    failures: list,
) -> None:
    """Flag a column with values outside the configured [min_val, max_val] range."""
    if col not in df.columns:
        failures.append(f"{col}: column missing from batch")
        return
    values = df[col].dropna()
    if min_val is not None and (values < min_val).any():
        failures.append(f"{col}: contains values below minimum {min_val}")
    if max_val is not None and (values > max_val).any():
        failures.append(f"{col}: contains values above maximum {max_val}")


def add_bounded_expectations(
    ge_suite: dict,
    feature_bounds_dict: dict[str, tuple[float, float | None]],
) -> dict:
    """
    Add per-column min/max range expectations to the validation suite config.

    Parameters
    ----------
    ge_suite : dict
        The suite config to extend (see save_checkpoint_config's checkpoint dict).
    feature_bounds_dict : dict[str, tuple[float, float | None]]
        Mapping of feature name to (min, max) bounds. None means unbounded
        on that side.

    Returns
    -------
    ge_suite : dict
        The suite config with bounded expectations added.
    """
    ge_suite.setdefault("expectations", [])
    for col, (min_val, max_val) in feature_bounds_dict.items():
        ge_suite["expectations"].append(
            {"column": col, "checks": [f"bounded_min_{min_val}_max_{max_val}"]}
        )
    return ge_suite


def _check_null_rate(df: pd.DataFrame, col: str, max_null_rate: float, failures: list) -> None:
    """Flag a column whose null rate exceeds max_null_rate."""
    if col not in df.columns:
        failures.append(f"{col}: column missing from batch")
        return
    null_rate = float(df[col].isna().mean())
    if null_rate > max_null_rate:
        failures.append(
            f"{col}: null rate {null_rate:.4%} exceeds maximum {max_null_rate:.4%}"
        )


def _check_row_count(df: pd.DataFrame, min_rows: int, max_rows: int, failures: list) -> None:
    """Flag a batch whose row count falls outside the configured bounds."""
    n = len(df)
    if not (min_rows <= n <= max_rows):
        failures.append(
            f"row_count: {n} outside expected range [{min_rows}, {max_rows}]"
        )


# ============= Suite Runner =============


def validate_batch(
    batch_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """
    Run the bounded network flow expectation suite against a feature batch.

    Expectations
    ------------
    - each feature column null rate <= MAX_NULL_RATE (2%)
    - table row count between MIN_ROW_COUNT and MAX_ROW_COUNT
    - each feature column value within its FEATURE_BOUNDS range

    Parameters
    ----------
    batch_df : pd.DataFrame
        Feature batch to validate.
    feature_cols : list[str] or None
        Feature columns to apply null-rate assertions to. Defaults to
        NUMERICAL_FEATURE_COLS.

    Returns
    -------
    o_passed : bool
        True if all expectations pass.
    failures : list of str
        Human-readable description of each failed expectation.
    """
    feature_cols = feature_cols if feature_cols is not None else NUMERICAL_FEATURE_COLS
    failures: list[str] = []

    _check_row_count(batch_df, MIN_ROW_COUNT, MAX_ROW_COUNT, failures)
    for col in feature_cols:
        _check_null_rate(batch_df, col, MAX_NULL_RATE, failures)
    for col, (min_val, max_val) in FEATURE_BOUNDS.items():
        _check_bounds(batch_df, col, min_val, max_val, failures)

    o_passed = len(failures) == 0

    if o_passed:
        print(f"Data validation [{SUITE_NAME}]: all expectations passed.")
    else:
        print(f"Data validation [{SUITE_NAME}] FAILED: {len(failures)} expectation(s) violated:")
        for f in failures:
            print(f"  - {f}")

    return o_passed, failures


def run_validation_suite(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Compatibility wrapper used by the Prefect retraining flow.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe to validate.

    Returns
    -------
    o_passed : bool
        True if all expectations pass.
    failures : list of str
        Description of each failed expectation.
    """
    return validate_batch(df)


# ============= Checkpoint =============


def save_checkpoint_config(output_dir: Path | None = None) -> None:
    """
    Save the validation checkpoint configuration to disk for documentation.

    Parameters
    ----------
    output_dir : Path or None
        Destination. Defaults to src/validation/checkpoints/.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "checkpoints"

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "name": "network_flow_checkpoint",
        "suite": SUITE_NAME,
        "expectations": [
            {"column": col, "checks": [f"null_rate_max_{MAX_NULL_RATE}"]}
            for col in NUMERICAL_FEATURE_COLS
        ]
        + [{"table": "row_count", "checks": [f"between_{MIN_ROW_COUNT}_{MAX_ROW_COUNT}"]}],
    }
    checkpoint = add_bounded_expectations(checkpoint, FEATURE_BOUNDS)

    path = output_dir / "network_flow_checkpoint.json"
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"Checkpoint config saved: {path}")


# ============= Main =============


if __name__ == "__main__":
    from src.config import TRAIN_FILE

    save_checkpoint_config()

    if TRAIN_FILE.exists():
        df = pd.read_parquet(TRAIN_FILE)
        o_passed, failures = run_validation_suite(df)
        if not o_passed:
            raise SystemExit(1)
    else:
        print("Training data not found. Run preprocessing first.")
