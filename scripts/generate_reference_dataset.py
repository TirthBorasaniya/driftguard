"""Generate the Evidently AI reference feature distribution from benign CICIDS2017 traffic."""

import sys
from pathlib import Path

# allow running as a standalone script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CICIDS_COLUMN_MAP, CICIDS_DATA_DIR
from src.features.engineering import FEATURE_COLS, compute_features_batch
from src.producer.flow_producer import load_cicids_file

# ============= Constants =============

REFERENCE_FILE = "Monday-WorkingHours.pcap_ISCX.csv"
REFERENCE_N_SAMPLES = 10_000
REFERENCE_OUTPUT_PATH = "data/reference/reference_network_flows.parquet"


# ============= Reference Generation =============


def generate_reference_dataset(
    data_dir: str,
    reference_file: str,
    column_map: dict[str, str],
    feature_cols: list[str],
    n_samples: int,
    output_path: str,
) -> None:
    """
    Sample benign records from the CICIDS2017 Monday file to build the Evidently
    AI reference feature distribution.

    Monday contains only benign traffic (no attacks), making it a clean baseline
    distribution for PSI drift computation against later attack days.

    Parameters
    ----------
    data_dir : str
        Directory containing CICIDS2017 CSV files.
    reference_file : str
        Filename of the file to sample from (Monday file, benign-only day).
    column_map : dict[str, str]
        Field mapping from raw CSV columns to schema fields.
    feature_cols : list[str]
        Feature columns to retain in the reference dataset.
    n_samples : int
        Number of records to sample for the reference distribution.
    output_path : str
        Path to write the reference Parquet file.
    """
    file_path = Path(data_dir) / reference_file
    if not file_path.exists():
        raise FileNotFoundError(
            f"Reference capture file not found: {file_path}\n"
            "Download CICIDS2017 from: https://www.unb.ca/cic/datasets/ids-2017.html"
        )

    raw_df = load_cicids_file(str(file_path), column_map)

    # retain benign rows only so the baseline is uncontaminated by attacks
    benign_df = raw_df[raw_df["label_binary"] == 0]

    feature_df = compute_features_batch(benign_df)[feature_cols]

    take = min(n_samples, len(feature_df))
    # fixed seed for a reproducible reference distribution
    reference_df = feature_df.sample(n=take, random_state=42).reset_index(drop=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    reference_df.to_parquet(out, index=False)
    print(f"Reference dataset written: {out} ({len(reference_df):,} rows, {len(feature_cols)} features)")


if __name__ == "__main__":
    generate_reference_dataset(
        data_dir=str(CICIDS_DATA_DIR),
        reference_file=REFERENCE_FILE,
        column_map=CICIDS_COLUMN_MAP,
        feature_cols=FEATURE_COLS,
        n_samples=REFERENCE_N_SAMPLES,
        output_path=REFERENCE_OUTPUT_PATH,
    )
