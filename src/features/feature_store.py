"""Feast feature store helpers for preparing, applying, and materializing features."""

import subprocess

import pandas as pd

from src.config import (
    FEAST_DIR,
    FEATURES_DIR,
    PROCESSED_DIR,
    TIME_COL,
)


# ============= Source Preparation =============


def prepare_feast_source():
    """
    Convert processed training CSV to parquet with proper timestamp for Feast.

    Feast FileSource requires parquet format and a datetime timestamp column.
    TransactionDT is converted from seconds-since-reference to datetime.

    Returns
    -------
    output_path : Path
        Path to the generated parquet file.
    """
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}. Run preprocessing first."
        )

    df = pd.read_csv(train_path)

    # convert TransactionDT (seconds since reference point) to datetime
    reference_date = pd.Timestamp("2017-12-01")
    df["event_timestamp"] = reference_date + pd.to_timedelta(df[TIME_COL], unit="s")
    df["created_timestamp"] = pd.Timestamp.now()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FEATURES_DIR / "transactions.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Feast source prepared: {output_path} ({len(df):,} rows)")

    return output_path


# ============= Feast Operations =============


def apply_feast():
    """
    Run feast apply from the feast directory to register feature definitions.

    Returns
    -------
    success : bool
        True if feast apply succeeded.
    """
    result = subprocess.run(
        ["feast", "apply"],
        cwd=str(FEAST_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"feast apply failed:\n{result.stderr}")
    print(result.stdout)
    return True


def materialize_features():
    """
    Materialize features to the online store for serving.

    Returns
    -------
    success : bool
        True if materialization succeeded.
    """
    from datetime import datetime

    end_date = datetime.now().isoformat()
    result = subprocess.run(
        ["feast", "materialize-incremental", end_date],
        cwd=str(FEAST_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"feast materialize failed:\n{result.stderr}")
    print(result.stdout)
    return True


# ============= Main =============


if __name__ == "__main__":
    prepare_feast_source()
    apply_feast()
    materialize_features()
