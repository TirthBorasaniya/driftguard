"""Tests for the percentile-bounds window scan behind DECISIONS D-022."""

import numpy as np
import pandas as pd
import pytest

from driftguard.benchmark.gx_percentile import count_windows, month_windows, scan_windows
from driftguard.data.registry import feature_names
from driftguard.training.validate import GXValidator, fit_bounds, physical_bounds

pytestmark = pytest.mark.unit

DAY_MS = 86_400_000


def frame(n, scale=1.0, seed=0, start_ms=0):
    rng = np.random.default_rng(seed)
    names = feature_names("cicids2017")
    values = scale * rng.lognormal(0.0, 1.0, size=(n, len(names)))
    out = pd.DataFrame(values, columns=names)
    out["event_time"] = start_ms + np.arange(n, dtype=np.int64)
    return out, names


def validators(reference, names):
    percentile = GXValidator(fit_bounds(reference, names), 0.99, 100, name="pct")
    physical = GXValidator(physical_bounds("cicids2017"), 1.0, 100, name="phys")
    return percentile, physical


def test_count_windows_yields_only_full_windows():
    data, names = frame(2_500)
    windows = list(count_windows(data, names, 1_000))
    assert [w[0] for w in windows] == ["window_00000", "window_00001"]
    assert windows[0][1].shape == (1_000, len(names)) and windows[1][2] == 1_999


def test_month_windows_group_by_calendar_month():
    data, names = frame(60)
    data["event_time"] = [1_136_073_600_000 + i * 2 * DAY_MS for i in range(60)]
    windows = list(month_windows(data, names))
    assert [w[0] for w in windows][:2] == ["2006-01", "2006-02"]
    assert sum(len(w[1]) for w in windows) == 60


def test_scan_counts_percentile_failures_that_physical_bounds_accept():
    reference, names = frame(50_000, seed=1)
    percentile, physical = validators(reference, names)
    data = pd.concat(
        [frame(1_000, seed=2, start_ms=0)[0], frame(1_000, scale=3.0, seed=3, start_ms=1_000)[0]],
        ignore_index=True,
    )
    summary = scan_windows(
        count_windows(data, names, 1_000), names, percentile, physical, reference_end_ms=999
    )
    assert summary["n_windows"] == 2 and summary["n_reference_windows"] == 1
    assert [w["percentile_failed"] for w in summary["windows"]] == [False, True]
    assert summary["n_physical_failed"] == 0
    assert summary["n_failed_percentile_only"] == 1
    assert summary["n_percentile_failed_after_reference"] == 1
    assert summary["first_percentile_failure_after_reference"] == "window_00001"
    assert summary["windows"][1]["percentile_features_outside"] > 0


def test_scan_flags_broken_data_under_both_suites():
    reference, names = frame(50_000, seed=1)
    percentile, physical = validators(reference, names)
    broken, _ = frame(1_000, seed=4)
    broken.loc[5, names[3]] = np.nan
    summary = scan_windows(count_windows(broken, names, 1_000), names, percentile, physical)
    assert summary["n_physical_failed"] == 1 and summary["n_percentile_failed"] == 1
    assert summary["n_failed_percentile_only"] == 0 and summary["n_reference_windows"] == 0
