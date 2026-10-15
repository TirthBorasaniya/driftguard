"""Promotion gate: paired bootstrap of the PR-AUC difference on shadow traffic (SPEC 7.1)."""

from dataclasses import asdict, dataclass

import numpy as np

# ============= Vectorized PR-AUC =============


def average_precision_rows(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Average precision for every row of a resample matrix, with tie handling, in NumPy.

    Matches `sklearn.metrics.average_precision_score`: rows are sorted by descending score,
    tied scores form one threshold, and AP = sum over thresholds of (R_k - R_{k-1}) * P_k.

    Parameters
    ----------
    labels : np.ndarray
        Binary labels, shape (n_resamples, n).
    scores : np.ndarray
        Scores, shape (n_resamples, n).

    Returns
    -------
    ap : np.ndarray
        Shape (n_resamples,); NaN for rows without positives.
    """
    order = np.argsort(-scores, axis=1, kind="stable")
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    sorted_labels = np.take_along_axis(labels, order, axis=1).astype(np.float64)
    tp = np.cumsum(sorted_labels, axis=1)
    rank = np.arange(1, scores.shape[1] + 1, dtype=np.float64)
    precision = tp / rank
    # last position of each group of tied scores is where a threshold takes effect
    is_end = np.ones_like(sorted_scores, dtype=bool)
    is_end[:, :-1] = sorted_scores[:, :-1] != sorted_scores[:, 1:]
    tp_at_ends = np.where(is_end, tp, 0.0)
    previous_end_tp = np.zeros_like(tp)
    previous_end_tp[:, 1:] = np.maximum.accumulate(tp_at_ends, axis=1)[:, :-1]
    n_pos = tp[:, -1]
    with np.errstate(invalid="ignore", divide="ignore"):
        contribution = np.where(is_end, (tp - previous_end_tp) * precision, 0.0)
        ap = contribution.sum(axis=1) / n_pos
    return np.where(n_pos > 0, ap, np.nan)


# ============= Bootstrap gate =============


@dataclass
class GateDecision:
    """Outcome of the shadow gate with its evidence."""

    decision: str
    delta_pr_auc: float
    ci_low: float
    ci_high: float
    champion_pr_auc: float
    challenger_pr_auc: float
    n_events: int
    n_positive: int
    n_resamples: int
    n_valid_resamples: int
    margin: float

    def as_dict(self) -> dict:
        """Plain dict for logs and results."""
        return asdict(self)


def paired_bootstrap_delta(
    labels: np.ndarray,
    champion: np.ndarray,
    challenger: np.ndarray,
    n_resamples: int,
    seed: int,
    chunk: int = 250,
) -> np.ndarray:
    """
    Paired bootstrap of `pr_auc(challenger) - pr_auc(champion)`.

    Each resample draws event indices with replacement once and evaluates both models on the
    same indices. Work is done in chunks of resamples to bound memory.

    Returns
    -------
    deltas : np.ndarray
        Shape (n_resamples,); NaN where a resample has no positives.
    """
    labels, champion, challenger = (np.asarray(a) for a in (labels, champion, challenger))
    rng = np.random.default_rng(seed)
    n = len(labels)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        idx = rng.integers(0, n, size=(size, n))
        y = labels[idx]
        deltas[start : start + size] = average_precision_rows(
            y, challenger[idx]
        ) - average_precision_rows(y, champion[idx])
    return deltas


def shadow_gate(
    labels: np.ndarray,
    champion: np.ndarray,
    challenger: np.ndarray,
    margin: float = 0.01,
    n_resamples: int = 1000,
    seed: int = 7,
) -> GateDecision:
    """
    Decide promote-to-canary or decline on the shadow-scored, labeled events.

    Promote if `delta_pr_auc >= margin` and the 95% bootstrap interval's lower bound is above 0;
    otherwise decline.
    """
    labels = np.asarray(labels).astype(int)
    if labels.sum() == 0:
        raise ValueError("shadow events contain no positives; the gate cannot compare PR-AUC")
    champ_ap = float(average_precision_rows(labels[None, :], np.asarray(champion)[None, :])[0])
    chal_ap = float(average_precision_rows(labels[None, :], np.asarray(challenger)[None, :])[0])
    delta = chal_ap - champ_ap
    deltas = paired_bootstrap_delta(labels, champion, challenger, n_resamples, seed)
    valid = deltas[~np.isnan(deltas)]
    low, high = (float(v) for v in np.percentile(valid, [2.5, 97.5]))
    decision = "promote" if delta >= margin and low > 0.0 else "decline"
    return GateDecision(
        decision,
        delta,
        low,
        high,
        champ_ap,
        chal_ap,
        len(labels),
        int(labels.sum()),
        n_resamples,
        len(valid),
        margin,
    )
