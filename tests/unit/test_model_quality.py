"""Regression tests for E1 validation splitting (a random slice leaked temporal neighbors)."""

import numpy as np
import pandas as pd
import pytest

from driftguard.benchmark.model_quality import blocked_validation_mask, split_validation

pytestmark = pytest.mark.unit


def frame(n=100_000):
    return pd.DataFrame({"label": (np.arange(n) % 7 == 0).astype(int), "x": np.arange(n)})


def test_blocked_validation_holds_out_whole_contiguous_blocks():
    mask = blocked_validation_mask(100_000, 5000, 0.1)
    assert abs(mask.mean() - 0.1) < 0.02
    edges = np.flatnonzero(np.diff(mask.astype(int)) != 0)
    runs = np.diff(np.concatenate([[-1], edges, [len(mask) - 1]]))
    assert runs[runs > 0].min() >= 5000 or len(runs) <= 2  # every run is at least one block


def test_blocked_split_has_no_row_adjacent_to_validation_inside_fit_blocks():
    data = frame()
    fit, val = split_validation(data, "blocked", 0.1, seed=0, block_size=5000)
    assert len(fit) + len(val) == len(data) and not set(fit.index) & set(val.index)
    # neighbors of validation rows are validation rows except at the 2 block edges
    in_val = np.zeros(len(data), dtype=bool)
    in_val[val.index.to_numpy()] = True
    boundary_pairs = int((in_val[1:] != in_val[:-1]).sum())
    assert boundary_pairs <= 2 * (len(val) // 5000 + 1)


def test_random_split_leaks_neighbors_unlike_blocked():
    data = frame()
    _, val_random = split_validation(data, "random_stratified", 0.1, seed=0)
    in_val = np.zeros(len(data), dtype=bool)
    in_val[val_random.index.to_numpy()] = True
    random_boundaries = int((in_val[1:] != in_val[:-1]).sum())
    _, val_blocked = split_validation(data, "blocked", 0.1, seed=0, block_size=5000)
    blocked = np.zeros(len(data), dtype=bool)
    blocked[val_blocked.index.to_numpy()] = True
    assert random_boundaries > 100 * int((blocked[1:] != blocked[:-1]).sum())


def test_blocked_split_restores_stream_order_for_shuffled_input():
    data = frame(20_000).sample(frac=1.0, random_state=1)
    fit, val = split_validation(data, "blocked", 0.1, seed=0, block_size=1000)
    assert fit.index.is_monotonic_increasing and val.index.is_monotonic_increasing
    assert val.index.min() == 9000  # first validation block is the 10th


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        split_validation(frame(100), "nope", 0.1, 0)
