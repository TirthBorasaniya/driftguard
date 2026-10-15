"""Drift detectors: Evidently drift share, score PSI, ADWIN and KSWIN wrappers (SPEC 5.2)."""

import logging
import warnings

import numpy as np
import pandas as pd

from driftguard.monitor.psi import bin_counts, compute_bin_edges, compute_psi

# ============= Evidently =============


class EvidentlyDriftShare:
    """
    Share of drifted columns from Evidently's data drift preset.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference features.
    max_rows : int
        Rows sampled per side to bound per-window cost.
    seed : int
        Sampling seed.
    """

    def __init__(self, reference: pd.DataFrame, max_rows: int = 5000, seed: int = 0):
        self.max_rows = max_rows
        self.seed = seed
        self.reference = self._sample(reference)

    def _sample(self, frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) <= self.max_rows:
            return frame.reset_index(drop=True)
        return frame.sample(self.max_rows, random_state=self.seed).reset_index(drop=True)

    def share(self, current: pd.DataFrame) -> float:
        """Return `share_of_drifted_columns` for the current window."""
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        with warnings.catch_warnings():
            # constant columns trigger harmless divide warnings inside Evidently's correlations
            warnings.simplefilter("ignore", category=RuntimeWarning)
            logging.getLogger("evidently").setLevel(logging.ERROR)
            report.run(reference_data=self.reference, current_data=self._sample(current))
        for metric in report.as_dict()["metrics"]:
            if metric["metric"] == "DatasetDriftMetric":
                return float(metric["result"]["share_of_drifted_columns"])
        raise RuntimeError("Evidently report has no DatasetDriftMetric")


# ============= Score PSI =============


class ScorePsi:
    """PSI between the champion's reference scores and the window's scores."""

    def __init__(self, reference_scores: np.ndarray, n_bins: int = 10, eps: float = 1e-6):
        self.edges = compute_bin_edges(reference_scores, n_bins)
        self.reference_counts = bin_counts(reference_scores, self.edges)
        self.eps = eps

    def psi(self, scores: np.ndarray) -> float:
        """Return the score-distribution PSI for one window."""
        return compute_psi(self.reference_counts, bin_counts(scores, self.edges), self.eps)


# ============= River change detectors =============


class AdwinErrorDetector:
    """
    ADWIN on the champion's 0/1 error stream from delayed labels (`river.drift.ADWIN`).

    Parameters
    ----------
    delta : float
        Confidence parameter (SPEC: 0.002).
    """

    def __init__(self, delta: float = 0.002):
        from river.drift import ADWIN

        self._factory = lambda: ADWIN(delta=delta)
        self.detector = self._factory()
        self.pending_detection = False

    def update(self, errors: list[int]) -> int:
        """Feed 0/1 errors; returns the number of detections in this call."""
        detections = 0
        for error in errors:
            self.detector.update(int(error))
            if self.detector.drift_detected:
                detections += 1
                self.pending_detection = True
        return detections

    def consume(self) -> float:
        """Return 1.0 if a change was detected since the last window close, then reset the flag."""
        fired, self.pending_detection = self.pending_detection, False
        return 1.0 if fired else 0.0

    def reset(self) -> None:
        """Restart the detector (after a promotion the error stream belongs to a new model)."""
        self.detector = self._factory()
        self.pending_detection = False


class KswinScoreDetector:
    """
    KSWIN on a subsampled champion score stream (`river.drift.KSWIN`).

    KSWIN runs a KS test on every update. Fed every event, a 5,000-event window holds thousands
    of overlapping tests at alpha 0.005 and fires in essentially every stationary window
    (measured: about 4 detections per 5,000 events in 40 of 40 windows). Its detection count is
    also no larger under a persistent shift than under the null, because it resets after each
    detection, so a count threshold cannot repair this. Testing one score in `stride` keeps the
    spec parameters and bounds the tests per window: with stride 100 about 5% of stationary
    windows are flagged, a strong shift is caught in the first window, and a mild one within two
    (measured on beta-distributed scores; see DECISIONS D-020).

    Parameters
    ----------
    alpha, window_size, stat_size : SPEC 5.2 defaults 0.005, 500, 100.
    stride : int
        Every `stride`-th score is fed to KSWIN; the offset persists across windows.
    seed : int
        Seed for KSWIN's sampling of the reference sub-window.
    """

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 500,
        stat_size: int = 100,
        seed: int = 42,
        stride: int = 100,
    ):
        from river.drift import KSWIN

        if stride < 1:
            raise ValueError("stride must be at least 1")
        self._factory = lambda: KSWIN(
            alpha=alpha, window_size=window_size, stat_size=stat_size, seed=seed
        )
        self.detector = self._factory()
        self.stride = int(stride)
        self._seen = 0

    def window_signal(self, scores: np.ndarray) -> tuple[float, int]:
        """
        Feed one window of scores (subsampled by `stride`).

        Returns
        -------
        signal : float
            1.0 if any change was detected inside the window, else 0.0.
        detections : int
            Number of detections.
        """
        detections = 0
        for score in np.asarray(scores, dtype=np.float64):
            self._seen += 1
            if self._seen % self.stride:
                continue
            self.detector.update(float(score))
            if self.detector.drift_detected:
                detections += 1
        return (1.0 if detections else 0.0), detections

    def reset(self) -> None:
        """Restart the detector."""
        self.detector = self._factory()
