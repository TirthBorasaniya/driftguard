"""Great Expectations validation suite: data quality checks run before every retraining cycle."""

import json
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, TARGET_COL


# ============= Expectation Definitions =============


def _check_no_nulls(df: pd.DataFrame, col: str, failures: list) -> None:
    null_count = df[col].isna().sum()
    if null_count > 0:
        failures.append(f"{col}: {null_count} null values found (expected 0)")


def _check_positive(df: pd.DataFrame, col: str, failures: list) -> None:
    non_positive = (df[col].dropna() <= 0).sum()
    if non_positive > 0:
        failures.append(f"{col}: {non_positive} non-positive values (expected all > 0)")


def _check_cardinality(
    df: pd.DataFrame,
    col: str,
    min_card: int,
    max_card: int,
    failures: list,
) -> None:
    card = df[col].nunique()
    if not (min_card <= card <= max_card):
        failures.append(
            f"{col}: cardinality {card} outside expected range [{min_card}, {max_card}]"
        )


def _check_label_rate(
    df: pd.DataFrame,
    col: str,
    min_rate: float,
    max_rate: float,
    failures: list,
) -> None:
    rate = df[col].mean()
    if not (min_rate <= rate <= max_rate):
        failures.append(
            f"{col}: fraud rate {rate:.4%} outside expected range "
            f"[{min_rate:.4%}, {max_rate:.4%}]"
        )


def _check_uniqueness(df: pd.DataFrame, col: str, failures: list) -> None:
    if col not in df.columns:
        return
    dup_count = df[col].duplicated().sum()
    if dup_count > 0:
        failures.append(f"{col}: {dup_count} duplicate values (expected 100% unique)")


# ============= Suite Runner =============


def run_validation_suite(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Run the fraud pipeline expectation suite against a dataframe.

    Expectations
    ------------
    - amt > 0, no nulls
    - city_pop > 0, no nulls
    - category cardinality between 10 and 20
    - state cardinality between 40 and 60
    - is_fraud rate between 0.3% and 2.0%
    - trans_num uniqueness = 100%

    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe to validate.

    Returns
    -------
    passed : bool
        True if all expectations pass.
    failures : list of str
        Human-readable description of each failed expectation.
    """
    failures = []

    _check_no_nulls(df, "amt", failures)
    _check_positive(df, "amt", failures)

    _check_no_nulls(df, "city_pop", failures)
    _check_positive(df, "city_pop", failures)

    _check_cardinality(df, "category", min_card=10, max_card=20, failures=failures)
    _check_cardinality(df, "state", min_card=40, max_card=60, failures=failures)

    _check_label_rate(df, TARGET_COL, min_rate=0.003, max_rate=0.02, failures=failures)

    _check_uniqueness(df, "trans_num", failures)

    passed = len(failures) == 0

    if passed:
        print("Data validation: all expectations passed.")
    else:
        print(f"Data validation FAILED: {len(failures)} expectation(s) violated:")
        for f in failures:
            print(f"  - {f}")

    return passed, failures


# ============= Checkpoint =============


def save_checkpoint_config(output_dir: Path = None) -> None:
    """
    Save GE checkpoint configuration to disk for documentation.

    Parameters
    ----------
    output_dir : Path or None
        Destination. Defaults to src/validation/checkpoints/.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "checkpoints"

    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "name": "fraud_checkpoint",
        "suite": "fraud_pipeline",
        "expectations": [
            {"column": "amt", "checks": ["no_nulls", "positive"]},
            {"column": "city_pop", "checks": ["no_nulls", "positive"]},
            {"column": "category", "checks": ["cardinality_10_to_20"]},
            {"column": "state", "checks": ["cardinality_40_to_60"]},
            {"column": "is_fraud", "checks": ["rate_0.003_to_0.02"]},
            {"column": "trans_num", "checks": ["uniqueness_100pct"]},
        ],
    }

    path = output_dir / "fraud_checkpoint.json"
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"Checkpoint config saved: {path}")


# ============= Main =============


if __name__ == "__main__":
    import pandas as pd
    from src.config import TRAIN_FILE

    save_checkpoint_config()

    if TRAIN_FILE.exists():
        df = pd.read_parquet(TRAIN_FILE)
        passed, failures = run_validation_suite(df)
        if not passed:
            raise SystemExit(1)
    else:
        print("Training data not found. Run preprocessing first.")
