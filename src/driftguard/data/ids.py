"""Deterministic event identifiers (SPEC 3.1)."""

import uuid

import numpy as np

# ============= Identifiers =============


def make_event_id(dataset: str, source_file: str, row_index: int) -> str:
    """
    Build the deterministic event id `uuid5(NAMESPACE_URL, "<dataset>:<source_file>:<row_index>")`.

    Parameters
    ----------
    dataset : str
        Dataset name, for example `cicids2017`.
    source_file : str
        Source file name the row came from.
    row_index : int
        Zero-based data row index in the source file, assigned before any cleaning.

    Returns
    -------
    event_id : str
        Canonical UUID string; identical across runs and processes.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{dataset}:{source_file}:{int(row_index)}"))


def make_event_ids(dataset: str, source_file: str, row_indices: np.ndarray) -> list[str]:
    """Vectorized wrapper over `make_event_id` for one source file."""
    prefix = f"{dataset}:{source_file}:"
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{prefix}{int(i)}")) for i in row_indices]
