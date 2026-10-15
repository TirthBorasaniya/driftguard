"""Synthetic flows with the dataset schema, for CI integration tests and smoke runs only."""

import numpy as np
import pandas as pd

from driftguard.data.ids import make_event_ids
from driftguard.data.registry import feature_names
from driftguard.data.schemas import DATASETS_WITH_IPS

# ============= Constants =============

SYNTHETIC_FILE = "synthetic.csv"


# ============= Generation =============


def synthetic_flows(
    dataset: str,
    n_events: int,
    attack_rate: float = 0.05,
    seed: int = 42,
    start_time_ms: int = 1_499_083_200_000,
    step_ms: int = 5,
    id_tag: str = "",
) -> pd.DataFrame:
    """
    Generate synthetic flows that follow the dataset's flow schema.

    Benign features are log-normal; attack rows are shifted so a model can separate them. The
    output is never used for reported results; it exists so integration tests run without the
    real datasets.

    Parameters
    ----------
    dataset : str
        Dataset whose feature list and schema to mimic.
    n_events : int
        Number of rows.
    attack_rate : float
        Fraction of rows labeled 1.
    seed : int
        Random seed.
    start_time_ms : int
        Event time of the first row.
    step_ms : int
        Event-time spacing between rows.
    id_tag : str
        Extra namespace for event ids so repeated test runs do not collide on idempotency keys.

    Returns
    -------
    frame : pd.DataFrame
        Columns matching the adapter output: identifiers, `event_time`, IPs (when the dataset
        carries them), features (float32), `label`, `label_family`.
    """
    rng = np.random.default_rng(seed)
    names = feature_names(dataset)
    labels = (rng.random(n_events) < attack_rate).astype("int8")
    values = rng.lognormal(mean=1.0, sigma=1.0, size=(n_events, len(names)))
    values[labels == 1] *= rng.uniform(2.0, 4.0, size=(int(labels.sum()), 1))
    frame = pd.DataFrame(values.astype("float32"), columns=names)
    row_index = np.arange(n_events, dtype=np.int64)
    meta = pd.DataFrame(
        {
            "event_id": make_event_ids(f"{dataset}-synthetic{id_tag}", SYNTHETIC_FILE, row_index),
            "dataset": dataset,
            "source_file": SYNTHETIC_FILE,
            "row_index": row_index,
            "event_time": start_time_ms + row_index * step_ms,
            "label": labels,
            "label_family": np.where(labels == 1, "synthetic_attack", "BENIGN"),
        }
    )
    if dataset in DATASETS_WITH_IPS:
        hosts = rng.integers(0, 256, size=(n_events, 2))
        meta["src_ip"] = [f"10.0.{a}.{b}" for a, b in hosts]
        meta["dst_ip"] = [f"10.1.{b}.{a}" for a, b in hosts]
    return pd.concat([meta, frame], axis=1)
