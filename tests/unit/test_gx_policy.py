"""Regression tests for D-022: drift must not be classified as broken data by the monitor."""

import numpy as np
import pandas as pd
import pytest

from driftguard.common.config import load_config
from driftguard.data.registry import dataset_config, feature_names
from driftguard.training.validate import GXValidator, fit_bounds, physical_bounds

pytestmark = pytest.mark.unit


def lognormal_frame(n, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    names = feature_names("cicids2017")[:8]
    return pd.DataFrame(scale * rng.lognormal(0.0, 1.0, size=(n, len(names))), columns=names)


def test_percentile_bounds_flag_a_covariate_shift_as_broken_data():
    """Records why the monitor cannot use fitted percentile bounds (E2 smoke: 100% blocked).

    Real flow features have heavier tails than this synthetic data, so a 1.2x shift was already
    enough there; here a 3x shift pushes more than 1% of values past the fitted 99.9th percentile.
    """
    reference = lognormal_frame(100_000, seed=1)
    names = list(reference.columns)
    percentile = GXValidator(fit_bounds(reference, names), mostly=0.99, min_rows=100, name="p")
    assert percentile.validate(lognormal_frame(5000, seed=2)).success
    assert not percentile.validate(lognormal_frame(5000, scale=3.0, seed=3)).success


def test_physical_bounds_pass_drift_and_reject_broken_data():
    names = feature_names("cicids2017")
    bounds = physical_bounds("cicids2017")
    quality = GXValidator(
        {n: bounds[n] for n in feature_names("cicids2017")[:8]}, mostly=1.0, min_rows=100, name="q"
    )
    drifted = lognormal_frame(5000, scale=3.0, seed=4)
    assert quality.validate(drifted).success
    bad = lognormal_frame(5000, seed=5)
    broken = {"negative": -1.0, "nan": np.nan, "inf": np.inf}
    for label, value in broken.items():
        damaged = bad.copy()
        damaged.iloc[10, 3] = value
        report = quality.validate(damaged)
        assert not report.success, label
    assert len(names) > 8 and bounds["dst_port"] == [0.0, 65535.0]
    assert bounds["fwd_header_length"][0] < -1e30  # no configured minimum: only non-finite fails


def test_gx_config_has_monitor_and_training_modes():
    for dataset in ("cicids2017", "kyoto"):
        gx = dataset_config(dataset)["gx"]
        assert gx["monitor_mostly"] == 1.0
        assert gx["training_bounds"] in ("percentile_bounds", "physical_bounds")


def test_training_validates_with_physical_bounds_by_decision_d022():
    for dataset in ("cicids2017", "kyoto"):
        assert dataset_config(dataset)["gx"]["training_bounds"] == "physical_bounds"
    assert load_config("configs/data/kyoto.yaml")["gx"]["training_bounds"] == "physical_bounds"
