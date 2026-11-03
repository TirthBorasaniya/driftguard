"""Tests for PSI, windows, the drift policy, detectors, and the monitor core."""

import fakeredis
import numpy as np
import pandas as pd
import pytest

from driftguard.common.config import load_config
from driftguard.monitor.detectors import AdwinErrorDetector, KswinScoreDetector, ScorePsi
from driftguard.monitor.main import PAUSE_KEY, PAUSED_BY_DATA_QUALITY, MonitorCore
from driftguard.monitor.policy import DriftPolicy
from driftguard.monitor.psi import PsiProfile, bin_counts, compute_bin_edges, compute_psi
from driftguard.monitor.reference import build_reference
from driftguard.monitor.windows import CountWindows, MonthWindows, month_id

pytestmark = pytest.mark.unit


def policy_config(**overrides):
    config = dict(load_config("configs/drift_policy.yaml").data)
    config.update(overrides)
    return config


# ----- PSI -----


def test_psi_identical_is_zero_and_known_value():
    assert compute_psi(np.array([10, 20, 30]), np.array([10, 20, 30])) == 0.0
    ref, cur = np.array([50, 50]), np.array([25, 75])
    expected = (0.25 - 0.5) * np.log(0.25 / 0.5) + (0.75 - 0.5) * np.log(0.75 / 0.5)
    assert compute_psi(ref, cur) == pytest.approx(expected)


def test_psi_floor_handles_empty_bins():
    value = compute_psi(np.array([100, 0]), np.array([0, 100]))
    assert np.isfinite(value) and value > 10


def test_bin_edges_quantiles_and_outside_values():
    ref = np.arange(1000, dtype=float)
    edges = compute_bin_edges(ref, 10)
    assert len(edges) == 11
    counts = bin_counts(np.array([-1e9, 1e9, 500.0]), edges)
    assert counts[0] == 1 and counts[-1] == 1 and counts.sum() == 3


def test_discrete_feature_collapses_bins_and_constant_is_zero():
    ref = np.repeat([0.0, 1.0, 2.0], 100)
    assert len(compute_bin_edges(ref, 10)) <= 4
    constant = np.full(100, 3.0)
    profile = PsiProfile.fit(constant[:, None], ["c"])
    assert profile.feature_psi(np.full((50, 1), 7.0))[0] == 0.0


def test_shifted_distribution_raises_psi():
    rng = np.random.default_rng(0)
    profile = PsiProfile.fit(rng.normal(size=(5000, 2)), ["a", "b"])
    same = profile.feature_psi(rng.normal(size=(5000, 2)))
    shifted = profile.feature_psi(rng.normal(loc=1.0, size=(5000, 2)))
    assert same.max() < 0.05 and shifted.min() > 0.25


# ----- windows -----


def test_count_windows_close_when_both_streams_full():
    windows = CountWindows(size=3)
    windows.add_flows([(i, [float(i)]) for i in range(7)])
    assert windows.ready() == []
    windows.add_scores([(i, 0.1) for i in range(6)])
    closed = windows.ready()
    assert [w.window_id for w in closed] == ["window_00000", "window_00001"]
    assert closed[1].features[:, 0].tolist() == [3.0, 4.0, 5.0]


def test_month_windows_close_after_later_events_and_count_late():
    jan, feb = 1_136_073_600_000, 1_138_752_000_000  # 2006-01-01, 2006-02-01 UTC
    assert month_id(jan) == "2006-01" and month_id(feb) == "2006-02"
    windows = MonthWindows(close_after=2)
    windows.add_flows([(jan, [1.0]), (jan, [2.0])])
    windows.add_scores([(jan, 0.1), (jan, 0.2)])
    windows.add_flows([(feb, [3.0])])
    windows.add_scores([(feb, 0.3)])
    assert windows.ready() == []
    windows.add_flows([(feb, [4.0])])
    windows.add_scores([(feb, 0.4)])
    closed = windows.ready()
    assert [w.window_id for w in closed] == ["2006-01"] and len(closed[0].scores) == 2
    windows.add_flows([(jan, [9.0])])
    assert windows.late_events == 1


# ----- policy -----


FLAG = {"psi_max": 0.3, "psi_share": 0.0, "drift_share": 0.0, "score_psi": 0.0}
CLEAN = {"psi_max": 0.01, "psi_share": 0.0, "drift_share": 0.0, "score_psi": 0.0}


def test_policy_debounce_requires_consecutive_flags():
    policy = DriftPolicy(policy_config())
    policy.add_labels(5000)
    assert not policy.evaluate("w0", FLAG, True).emit
    assert not policy.evaluate("w1", CLEAN, True).emit
    assert "debounce" in policy.evaluate("w2", FLAG, True).blocked_by
    decision = policy.evaluate("w3", FLAG, True)
    assert decision.emit and decision.severity == "LOW"


def test_policy_min_labeled_and_cooldown():
    policy = DriftPolicy(policy_config())
    policy.evaluate("w0", FLAG, True)
    assert "min_labeled" in policy.evaluate("w1", FLAG, True).blocked_by
    policy.add_labels(2000)
    assert policy.evaluate("w2", FLAG, True).emit
    blocked = [policy.evaluate(f"w{i}", FLAG, True) for i in range(3, 6)]
    assert all("cooldown" in d.blocked_by for d in blocked)
    assert policy.evaluate("w6", FLAG, True).emit


def test_policy_promotion_resets_and_gx_failure_never_emits():
    policy = DriftPolicy(policy_config())
    policy.add_labels(10_000)
    policy.note_promotion()
    assert policy.labeled_since_promotion == 0
    policy.add_labels(10_000)
    for i in range(3):
        policy.evaluate(f"c{i}", CLEAN, True)
    policy.evaluate("w0", FLAG, True)
    decision = policy.evaluate("w1", FLAG, False)
    assert decision.data_quality_failed and not decision.emit
    assert policy.consecutive == 0


def test_policy_severity_high_on_adwin_or_large_psi():
    policy = DriftPolicy(policy_config())
    assert policy.severity({"psi_max": 0.6}) == "HIGH"
    assert policy.severity({"psi_max": 0.1, "adwin_error": 1.0}) == "HIGH"
    assert policy.severity({"psi_max": 0.3}) == "LOW"
    assert policy.flag_reasons({"kswin_score": 1.0, "psi_max": 0.0}) == ["kswin_score"]


# ----- detectors -----


def test_adwin_detects_error_rate_step():
    detector = AdwinErrorDetector(0.002)
    rng = np.random.default_rng(1)
    assert detector.update((rng.random(2000) < 0.02).astype(int).tolist()) == 0
    assert detector.update((rng.random(2000) < 0.5).astype(int).tolist()) >= 1
    assert detector.consume() == 1.0 and detector.consume() == 0.0


def test_kswin_wrapper_detects_a_step_change_at_stride_one():
    rng = np.random.default_rng(2)
    detector = KswinScoreDetector(0.005, 500, 100, seed=1, stride=1)
    detector.window_signal(rng.beta(2, 8, 600))
    signal, detections = detector.window_signal(rng.beta(8, 2, 600))
    assert signal == 1.0 and detections >= 1


def test_score_psi():
    rng = np.random.default_rng(3)
    scorer = ScorePsi(rng.beta(2, 8, 5000))
    assert scorer.psi(rng.beta(2, 8, 5000)) < 0.05
    assert scorer.psi(rng.beta(8, 2, 5000)) > 0.2


# ----- monitor core -----


class FakeGX:
    def __init__(self, ok):
        self.ok = ok

    def validate(self, frame):
        from driftguard.training.validate import ValidationReport

        return ValidationReport(self.ok, [] if self.ok else ["x"], 1)


def _core(gx_ok=True, window_size=200):
    from driftguard.data.registry import feature_names

    names = feature_names("cicids2017")
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(rng.lognormal(size=(2000, len(names))), columns=names)
    ref["score"] = rng.beta(2, 8, 2000)
    redis = fakeredis.FakeRedis(decode_responses=True)
    core = MonitorCore(
        "cicids2017", policy_config(window_size=window_size, min_labeled=0), redis, FakeGX(gx_ok)
    )
    core.set_reference(build_reference("1", ref, names, 10, 500), o_promotion=False)
    return core, names, redis


def _flows(names, n, loc, seed):
    rng = np.random.default_rng(seed)
    values = rng.lognormal(mean=loc, size=(n, len(names)))
    return [
        {
            "event_id": f"e{seed}-{i}",
            "event_time": i,
            **{name: float(values[i, j]) for j, name in enumerate(names)},
        }
        for i in range(n)
    ]


def _preds(flows, score):
    return [
        {"event_id": f["event_id"], "event_time": f["event_time"], "score": score, "decision": 0}
        for f in flows
    ]


def test_monitor_emits_event_after_debounced_shift():
    core, names, _ = _core()
    events = []
    for k in range(4):
        flows = _flows(names, 200, 3.0, k)
        events += core.on_flows(flows) + core.on_predictions(_preds(flows, 0.9))
    assert len(events) == 1
    event = events[0]
    assert event["severity"] == "HIGH" and event["policy_version"] == 5
    assert event["signals"]["psi_max"] > 0.5 and "window_end_ms" in event


def test_monitor_pauses_on_gx_failure_and_resumes_on_clean_window():
    core, names, redis = _core(gx_ok=False)
    flows = _flows(names, 200, 0.0, 9)
    core.on_flows(flows)
    core.on_predictions(_preds(flows, 0.1))
    assert redis.get(PAUSE_KEY.format(dataset="cicids2017")) == PAUSED_BY_DATA_QUALITY
    core.gx = FakeGX(True)
    flows = _flows(names, 200, 0.0, 10)
    core.on_flows(flows)
    core.on_predictions(_preds(flows, 0.1))
    assert redis.get(PAUSE_KEY.format(dataset="cicids2017")) is None


def test_operator_pause_is_not_lifted():
    core, names, redis = _core()
    redis.set(PAUSE_KEY.format(dataset="cicids2017"), "operator")
    flows = _flows(names, 200, 0.0, 11)
    core.on_flows(flows)
    core.on_predictions(_preds(flows, 0.1))
    assert redis.get(PAUSE_KEY.format(dataset="cicids2017")) == "operator"


def test_labels_join_predictions_for_adwin():
    core, names, _ = _core()
    flows = _flows(names, 10, 0.0, 12)
    core.on_predictions(_preds(flows, 0.1))
    core.on_labels([{"event_id": f["event_id"], "label": 1} for f in flows])
    assert core.policy.labeled_since_promotion == 10
    assert not core._pending_predictions and not core._pending_labels


def test_kswin_stride_bounds_stationary_false_alarms_and_still_detects_shift():
    """Regression: per-event KSWIN tests fired in every stationary window (E2 smoke run)."""
    rng = np.random.default_rng(5)
    window = 5000
    per_event = KswinScoreDetector(0.005, 500, 100, seed=1, stride=1)
    per_event_flags = [per_event.window_signal(rng.beta(2, 8, window))[0] for _ in range(5)]
    assert sum(per_event_flags) >= 4  # the spec parameters on every event: constant false alarms

    flagged = 0
    for seed in range(3):
        strided = KswinScoreDetector(0.005, 500, 100, seed=seed, stride=100)
        for _ in range(14):  # 700 tests: the window fills after 500
            flagged += strided.window_signal(rng.beta(2, 8, window))[0] > 0
    assert flagged <= 6  # about 5 percent of 42 windows, never most of them

    detector = KswinScoreDetector(0.005, 500, 100, seed=2, stride=100)
    for _ in range(14):
        detector.window_signal(rng.beta(2, 8, window))
    signals = [detector.window_signal(rng.beta(8, 2, window))[0] for _ in range(3)]
    assert max(signals) == 1.0


def test_kswin_stride_offset_persists_across_windows():
    detector = KswinScoreDetector(stride=10)
    detector.window_signal(np.zeros(25))
    assert detector._seen == 25
    detector.window_signal(np.zeros(25))
    assert detector._seen == 50
    with pytest.raises(ValueError):
        KswinScoreDetector(stride=0)


def test_policy_version_five_carries_kswin_stride_and_latch_span():
    config = policy_config()
    assert config["version"] == 5 and config["kswin"]["stride"] == 100
    assert config["debounce_windows"] == 2


# ----- change-point latch (D-023) -----


def _ready_policy(**overrides):
    policy = DriftPolicy(policy_config(**overrides))
    policy.add_labels(10_000)
    return policy


def test_lone_adwin_detection_emits_after_the_debounce_span():
    policy = _ready_policy()
    first = policy.evaluate("w0", {**CLEAN, "adwin_error": 1.0}, True)
    assert first.flagged and first.reasons == ["adwin_error"] and not first.emit
    assert "debounce" in first.blocked_by
    second = policy.evaluate("w1", CLEAN, True)
    assert second.flagged and second.reasons == ["adwin_error"]
    assert second.emit and second.severity == "HIGH"


def test_lone_kswin_detection_emits_low_severity_after_the_debounce_span():
    policy = _ready_policy()
    assert not policy.evaluate("w0", {**CLEAN, "kswin_score": 1.0}, True).emit
    decision = policy.evaluate("w1", CLEAN, True)
    assert decision.emit and decision.severity == "LOW" and decision.reasons == ["kswin_score"]


def test_latch_expires_after_the_span_without_a_new_detection():
    policy = _ready_policy(debounce_windows=2, cooldown_windows=0)
    policy.evaluate("w0", {**CLEAN, "kswin_score": 1.0}, True)
    policy.evaluate("w1", CLEAN, True)  # emits and clears
    assert not policy.evaluate("w2", CLEAN, True).flagged
    assert not policy.evaluate("w3", CLEAN, True).flagged


def test_latch_span_follows_debounce_windows():
    policy = _ready_policy(debounce_windows=3)
    decisions = [policy.evaluate("w0", {**CLEAN, "adwin_error": 1.0}, True)]
    decisions += [policy.evaluate(f"w{i}", CLEAN, True) for i in (1, 2, 3)]
    assert [d.flagged for d in decisions] == [True, True, True, False]
    assert [d.emit for d in decisions] == [False, False, True, False]


def test_a_new_detection_refreshes_the_latch():
    policy = _ready_policy(debounce_windows=2, min_labeled=10**9)
    policy.evaluate("w0", {**CLEAN, "adwin_error": 1.0}, True)
    policy.evaluate("w1", {**CLEAN, "adwin_error": 1.0}, True)
    assert policy.evaluate("w2", CLEAN, True).flagged
    assert not policy.evaluate("w3", CLEAN, True).flagged


def test_latch_does_not_change_debounce_for_threshold_signals():
    policy = _ready_policy()
    assert not policy.evaluate("w0", FLAG, True).emit
    assert not policy.evaluate("w1", CLEAN, True).flagged
    assert not policy.evaluate("w2", CLEAN, True).emit


def test_gx_failure_still_ages_the_latch_and_never_emits():
    policy = _ready_policy()
    policy.evaluate("w0", {**CLEAN, "adwin_error": 1.0}, True)
    failed = policy.evaluate("w1", CLEAN, False)
    assert failed.data_quality_failed and not failed.emit and policy.consecutive == 0
    assert not policy.evaluate("w2", CLEAN, True).flagged


def test_promotion_clears_the_latch():
    policy = _ready_policy()
    policy.evaluate("w0", {**CLEAN, "kswin_score": 1.0}, True)
    policy.note_promotion()
    assert not policy.evaluate("w1", CLEAN, True).flagged


def test_evaluate_does_not_mutate_the_callers_signals():
    policy = _ready_policy()
    signals = {**CLEAN, "adwin_error": 1.0}
    policy.evaluate("w0", signals, True)
    follow_up = dict(CLEAN)
    policy.evaluate("w1", follow_up, True)
    assert signals["adwin_error"] == 1.0 and "adwin_error" not in follow_up
