"""Tests for the drift injector and the E2 benchmark mechanics."""

import numpy as np
import pandas as pd
import pytest

from driftguard.benchmark.drift_benchmark import (
    label_arrivals,
    prepare_base,
    run_dataset,
    score_flags,
    summarize,
)
from driftguard.benchmark.injector import inject
from driftguard.common.config import load_config
from driftguard.data.synthetic import synthetic_flows

pytestmark = pytest.mark.unit


def _segment(n=1000, seed=0):
    return synthetic_flows("cicids2017", n, attack_rate=0.1, seed=seed)


def test_covariate_shift_only_after_onset_and_selected_features(tmp_path):
    seg = _segment()
    shifted, manifest = inject(
        seg,
        "covariate",
        400,
        {"features": ["flow_duration"], "factor": 2.0, "noise_std": 0.0},
        seed=1,
        reference=seg,
    )
    assert np.allclose(shifted["flow_duration"][:400], seg["flow_duration"][:400])
    assert np.allclose(shifted["flow_duration"][400:], 2.0 * seg["flow_duration"][400:], rtol=1e-6)
    assert np.allclose(shifted["protocol"], seg["protocol"])
    assert manifest.onset == 400 and manifest.rows_modified == 600
    manifest.write(tmp_path / "m.json")
    assert '"shift_type": "covariate"' in (tmp_path / "m.json").read_text()


def test_prior_shift_reaches_target_attack_rate():
    seg = _segment(4000)
    pool = seg[seg["label"] == 1]
    shifted, _ = inject(
        seg, "prior", 1000, {"attack_rate_after": 0.2}, seed=2, reference=seg, attack_pool=pool
    )
    assert len(shifted) == len(seg)
    assert shifted["label"].iloc[1000:].mean() == pytest.approx(0.2, abs=0.001)
    assert shifted["label"].iloc[:1000].equals(seg["label"].iloc[:1000])


def test_concept_shift_flips_region_after_onset_without_feature_change():
    seg = _segment(2000)
    rule = {"feature": "flow_duration", "quantile": 0.5}
    shifted, manifest = inject(
        seg, "concept", 1000, {"region_rule": rule, "flip_fraction": 0.5}, seed=3, reference=seg
    )
    changed = shifted["label"] != seg["label"]
    assert not changed.iloc[:1000].any() and manifest.rows_modified == int(changed.sum()) > 0
    threshold = np.quantile(seg["flow_duration"], 0.5)
    assert (shifted.loc[changed, "flow_duration"] <= threshold).all()
    assert shifted[[c for c in seg.columns if c not in ("label", "label_family")]].equals(
        seg[[c for c in seg.columns if c not in ("label", "label_family")]]
    )


def test_invalid_shift_inputs():
    seg = _segment(100)
    with pytest.raises(ValueError):
        inject(seg, "label_noise", 10, {}, 0, seg)
    with pytest.raises(ValueError):
        inject(seg, "covariate", 0, {"features": [], "factor": 1, "noise_std": 0}, 0, seg)


def test_score_flags_and_summary():
    flags = [False, True, False, False, False, True, True]
    scored = score_flags(flags, onset_window=4, horizon=10)
    assert scored == {"delay": 1, "missed": False, "false_alarms": 1, "pre_windows": 4}
    missed = score_flags([False] * 20, onset_window=5, horizon=3)
    assert missed["missed"] and missed["delay"] is None
    late = score_flags([False] * 9 + [True], onset_window=2, horizon=3)
    assert late["missed"]
    summary = summarize([scored, missed])
    assert summary["miss_rate"] == 0.5 and summary["median_delay_windows"] == 1.0
    assert summary["false_alarms_per_100_windows"] == pytest.approx(100 * 1 / 9)


def test_label_arrivals_coverage_and_delay():
    arrival = label_arrivals(10_000, 0.3, 5, 10, np.random.default_rng(0))
    covered = arrival >= 0
    assert abs(covered.mean() - 0.3) < 0.02
    delays = arrival[covered] - np.flatnonzero(covered)
    assert delays.min() >= 5 and delays.max() <= 10


def test_small_benchmark_detects_strong_covariate_shift():
    frame = synthetic_flows("cicids2017", 12_000, attack_rate=0.1, seed=5)
    policy = dict(load_config("configs/drift_policy.yaml").data)
    policy["min_labeled"] = 0
    model = dict(load_config("configs/model.yaml").data)
    model["lightgbm"] = {**model["lightgbm"], "n_estimators": 20, "n_jobs": 1}
    config = {
        "window_size": 300,
        "n_windows": 12,
        "onset_window_range": [4, 6],
        "miss_horizon_windows": 4,
        "n_seeds": 2,
        "chunk_size": 100,
        "train_fraction": 0.3,
        "label_coverages": [1.0],
        "label_delay": {"min": 10, "max": 50},
        "shifts": {
            "covariate": {
                "factors": [3.0],
                "noise_std": 0.0,
                "features": {"cicids2017": ["flow_duration", "flow_iat_mean"]},
            },
            "prior": {"attack_rate_after": [0.5]},
            "concept": {
                "flip_fraction": 0.5,
                "region_rule": {"cicids2017": {"feature": "flow_duration", "quantile": 0.5}},
            },
        },
    }
    policy["evidently_max_rows"] = 300
    base = prepare_base("cicids2017", frame, config, model, policy)
    results = run_dataset(base, config, policy)["label_coverage=1.0"]
    psi = results["covariate:factor=3.0"]["detectors"]["psi_max"]
    assert psi["miss_rate"] == 0.0 and psi["median_delay_windows"] <= 1
    gx = results["covariate:factor=3.0"]["gx"]
    assert set(gx) == {
        "failures_per_100_windows_before_onset",
        "share_of_windows_blocked_after_onset",
    }
    # the monitor's suite checks data quality, not distribution: a x3 covariate shift is drift,
    # so it must not be blocked as broken data (it was, at every factor, with percentile bounds)
    assert gx["share_of_windows_blocked_after_onset"] == 0.0
    assert gx["failures_per_100_windows_before_onset"] == 0.0
    policy = results["covariate:factor=3.0"]["detectors"]["policy"]
    assert policy["miss_rate"] == 0.0
    assert set(results) == {
        "covariate:factor=3.0",
        "prior:attack_rate_after=0.5",
        "concept:flip_fraction=0.5",
    }
    assert isinstance(pd.Series([1]).sum(), np.integer)


def test_summarize_gx_counts():
    from driftguard.benchmark.drift_benchmark import summarize_gx

    runs = [
        {
            "failures_before_onset": 1,
            "windows_before_onset": 10,
            "blocked_after_onset": 3,
            "windows_after_onset": 6,
        },
        {
            "failures_before_onset": 0,
            "windows_before_onset": 10,
            "blocked_after_onset": 0,
            "windows_after_onset": 6,
        },
    ]
    summary = summarize_gx(runs)
    assert summary["failures_per_100_windows_before_onset"] == 5.0
    assert summary["share_of_windows_blocked_after_onset"] == 0.25
    assert summarize_gx([])["failures_per_100_windows_before_onset"] is None
