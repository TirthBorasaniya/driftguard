"""Tests for PSI-specific drift trigger (infra improvement 2)."""

import pytest

pytest.importorskip("evidently")

from src.monitoring.drift_detector import (  # noqa: E402
    PSI_MINOR_THRESHOLD,
    PSI_SIGNIFICANT_THRESHOLD,
    check_drift_and_trigger_retraining,
)


def test_triggers_retraining_above_significant_threshold():
    calls = []
    triggered = check_drift_and_trigger_retraining(
        psi_score=PSI_SIGNIFICANT_THRESHOLD + 0.01,
        significant_threshold=PSI_SIGNIFICANT_THRESHOLD,
        retraining_flow_fn=lambda: calls.append(1),
    )
    assert triggered is True
    assert len(calls) == 1


def test_does_not_trigger_below_minor_threshold():
    calls = []
    triggered = check_drift_and_trigger_retraining(
        psi_score=PSI_MINOR_THRESHOLD - 0.01,
        significant_threshold=PSI_SIGNIFICANT_THRESHOLD,
        retraining_flow_fn=lambda: calls.append(1),
    )
    assert triggered is False
    assert len(calls) == 0
