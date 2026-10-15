"""Tests for E4 aggregation helpers and the static figures."""

import numpy as np
import pandas as pd
import pytest

from driftguard.benchmark.adaptation import (
    per_period_pr_auc,
    rolling_pr_auc,
    summarize_lifecycle,
    wall_to_event_time,
)
from driftguard.benchmark.figures import cicids_timeline, kyoto_per_year, replica_timeline

pytestmark = pytest.mark.unit


def _frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.2).astype(int)
    return pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(n)],
            "event_time": np.arange(n) * 1000,
            "label": labels,
            "score": rng.random(n) + labels,
        }
    )


def test_period_and_rolling_pr_auc():
    frame = _frame()
    periods = per_period_pr_auc(frame, pd.Series(np.where(frame.index < 500, 2008, 2009)))
    assert set(periods) == {"2008", "2009"} and periods["2008"]["n"] == 500
    assert 0.5 < periods["2009"]["pr_auc"] <= 1.0
    rolling = rolling_pr_auc(frame, 300)
    assert [r["n"] for r in rolling] == [300, 300, 300, 100]
    empty = per_period_pr_auc(frame.assign(label=0), pd.Series([1] * len(frame)))
    assert empty["1"]["pr_auc"] is None


def test_lifecycle_counts_and_time_mapping():
    promotions = [
        {"action": a}
        for a in ("promote_to_canary", "promote", "decline", "canary_rolled_back", "rollback")
    ]
    drift = [{"severity": "HIGH"}, {"severity": "LOW"}]
    assert summarize_lifecycle(promotions, drift) == {
        "drift_events": 2,
        "drift_events_high": 1,
        "promotions": 1,
        "promoted_to_canary": 1,
        "declines": 1,
        "rollbacks": 2,
    }
    progress = [(1_700_000_000_000, 10), (1_700_000_060_000, 20)]
    assert wall_to_event_time("2023-11-14T22:13:30.000000Z", progress) == 10
    assert wall_to_event_time("2023-11-14T22:14:30.000000Z", progress) == 20
    assert wall_to_event_time("2020-01-01T00:00:00.000000Z", progress) is None


def test_figures_render(tmp_path):
    years = {
        "static": {"2008": {"pr_auc": 0.6}, "2009": {"pr_auc": 0.5}},
        "adaptive": {"2008": {"pr_auc": 0.6}, "2009": {"pr_auc": 0.7}},
    }
    assert kyoto_per_year(years, tmp_path / "k.png").stat().st_size > 1000
    rolling = rolling_pr_auc(_frame(), 200)
    assert cicids_timeline(
        rolling, [300_000.0], [(500_000.0, "decline")], tmp_path / "c.png"
    ).exists()
    assert replica_timeline(
        [{"t_s": 0, "replicas": 1}, {"t_s": 60, "replicas": 4}], tmp_path / "r.png"
    ).exists()
