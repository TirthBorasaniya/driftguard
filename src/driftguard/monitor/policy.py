"""Versioned drift decision policy: flags, debounce, min_labeled, cooldown, severity (SPEC 5.3)."""

from dataclasses import dataclass, field
from typing import Any

# ============= Constants =============

THRESHOLD_SIGNALS = ("psi_max", "psi_share", "drift_share", "score_psi")
CHANGE_SIGNALS = ("kswin_score", "adwin_error")
LOW, HIGH = "LOW", "HIGH"


# ============= Decision =============


@dataclass
class PolicyDecision:
    """Outcome of evaluating one window."""

    window_id: str
    flagged: bool
    reasons: list[str]
    emit: bool = False
    severity: str | None = None
    data_quality_failed: bool = False
    blocked_by: list[str] = field(default_factory=list)


class DriftPolicy:
    """
    Stateful policy over successive windows.

    A window is flagged if any unsupervised signal crosses its threshold or a change detector
    fired (signal value 1.0). A drift event is emitted after `debounce_windows` consecutive
    flagged windows, once at least `min_labeled` labels arrived since the last promotion, and
    after `cooldown_windows` windows since the last event or promotion. A window failing the GX
    suite never emits and resets the debounce count; retraining pauses until a clean window.

    Parameters
    ----------
    config : dict
        `configs/drift_policy.yaml` content.
    """

    def __init__(self, config: dict[str, Any]):
        self.version = int(config["version"])
        self.thresholds = dict(config["thresholds"])
        self.debounce = int(config["debounce_windows"])
        self.cooldown = int(config["cooldown_windows"])
        self.min_labeled = int(config["min_labeled"])
        self.high_psi = float(config["severity_high"]["psi_max"])
        self.high_on_adwin = bool(config["severity_high"]["adwin"])
        self.consecutive = 0
        self.windows_since = self.cooldown
        self.labeled_since_promotion = 0

    def add_labels(self, n: int) -> None:
        """Count labels received since the last promotion."""
        self.labeled_since_promotion += int(n)

    def note_promotion(self) -> None:
        """Reset counters when a new champion is promoted."""
        self.labeled_since_promotion = 0
        self.windows_since = 0
        self.consecutive = 0

    def flag_reasons(self, signals: dict[str, float]) -> list[str]:
        """Return the names of the signals that flag this window."""
        reasons = [
            name
            for name in THRESHOLD_SIGNALS
            if name in signals and signals[name] > self.thresholds[name]
        ]
        reasons += [name for name in CHANGE_SIGNALS if signals.get(name, 0.0) >= 1.0]
        return reasons

    def severity(self, signals: dict[str, float]) -> str:
        """HIGH if ADWIN fired or psi_max exceeds the high threshold; otherwise LOW."""
        if self.high_on_adwin and signals.get("adwin_error", 0.0) >= 1.0:
            return HIGH
        if signals.get("psi_max", 0.0) > self.high_psi:
            return HIGH
        return LOW

    def evaluate(self, window_id: str, signals: dict[str, float], o_gx_ok: bool) -> PolicyDecision:
        """
        Evaluate one closed window and update state.

        Parameters
        ----------
        window_id : str
            Window identifier.
        signals : dict
            Signal values for the window.
        o_gx_ok : bool
            False if the GX suite failed on the window.

        Returns
        -------
        decision : PolicyDecision
        """
        self.windows_since += 1
        reasons = self.flag_reasons(signals)
        decision = PolicyDecision(window_id, flagged=bool(reasons), reasons=reasons)
        if not o_gx_ok:
            decision.data_quality_failed = True
            decision.blocked_by.append("gx_failure")
            self.consecutive = 0
            return decision
        self.consecutive = self.consecutive + 1 if reasons else 0
        if not reasons:
            return decision
        if self.consecutive < self.debounce:
            decision.blocked_by.append("debounce")
        if self.labeled_since_promotion < self.min_labeled:
            decision.blocked_by.append("min_labeled")
        if self.windows_since <= self.cooldown:
            decision.blocked_by.append("cooldown")
        if decision.blocked_by:
            return decision
        decision.emit = True
        decision.severity = self.severity(signals)
        self.consecutive = 0
        self.windows_since = 0
        return decision
