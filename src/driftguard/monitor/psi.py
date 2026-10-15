"""Population stability index with reference-quantile bins (SPEC 5.2)."""

from dataclasses import dataclass

import numpy as np

# ============= Constants =============

DEFAULT_FLOOR = 1e-6


# ============= Binning =============


def compute_bin_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Quantile bin edges from a reference sample.

    Duplicate quantiles (discrete or discretized features, such as the AnoShift Kyoto bins)
    collapse, so the number of bins can be smaller than `n_bins` and aligns with the
    discretization levels.

    Parameters
    ----------
    reference : np.ndarray
        Reference values, shape (n,).
    n_bins : int
        Target number of bins.

    Returns
    -------
    edges : np.ndarray
        Sorted unique edges, shape (k + 1,) with k <= n_bins; length 1 for a constant feature.
    """
    reference = np.asarray(reference, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        raise ValueError("reference sample is empty")
    return np.unique(np.quantile(reference, np.linspace(0.0, 1.0, n_bins + 1)))


def bin_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Count values per bin; values outside the reference range fall into the edge bins.

    Returns
    -------
    counts : np.ndarray
        int64 counts, shape (max(len(edges) - 1, 1),).
    """
    n_bins = max(len(edges) - 1, 1)
    inner = edges[1:-1]
    index = np.searchsorted(inner, np.asarray(values, dtype=np.float64), side="right")
    return np.bincount(np.clip(index, 0, n_bins - 1), minlength=n_bins).astype(np.int64)


def compute_psi(
    reference_counts: np.ndarray, current_counts: np.ndarray, eps: float = DEFAULT_FLOOR
) -> float:
    """
    Compute the population stability index between two binned distributions.

    Uses `sum((p_cur - p_ref) * ln(p_cur / p_ref))` with proportions floored at `eps`.

    Parameters
    ----------
    reference_counts : np.ndarray
        Bin counts of the reference window, shape (n_bins,).
    current_counts : np.ndarray
        Bin counts of the current window, shape (n_bins,).
    eps : float
        Floor applied to bin proportions to avoid division by zero.

    Returns
    -------
    psi : float
        Population stability index, non-negative.
    """
    ref = np.asarray(reference_counts, dtype=np.float64)
    cur = np.asarray(current_counts, dtype=np.float64)
    if ref.shape != cur.shape:
        raise ValueError("count arrays must have the same shape")
    if ref.sum() == 0 or cur.sum() == 0:
        raise ValueError("count arrays must be non-empty")
    p_ref = np.maximum(ref / ref.sum(), eps)
    p_cur = np.maximum(cur / cur.sum(), eps)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


# ============= Reference profile =============


@dataclass
class PsiProfile:
    """
    Per-feature bin edges and reference counts.

    Attributes
    ----------
    names : list of str
        Feature names in column order.
    edges : list of np.ndarray
        Edges per feature, each shape (k_i + 1,).
    reference_counts : list of np.ndarray
        Reference counts per feature, each shape (k_i,).
    """

    names: list[str]
    edges: list[np.ndarray]
    reference_counts: list[np.ndarray]

    @classmethod
    def fit(cls, reference: np.ndarray, names: list[str], n_bins: int = 10) -> "PsiProfile":
        """Fit a profile on a reference matrix of shape (n, len(names))."""
        reference = np.asarray(reference, dtype=np.float64)
        edges = [compute_bin_edges(reference[:, j], n_bins) for j in range(len(names))]
        counts = [bin_counts(reference[:, j], e) for j, e in enumerate(edges)]
        return cls(list(names), edges, counts)

    def feature_psi(self, current: np.ndarray, eps: float = DEFAULT_FLOOR) -> np.ndarray:
        """PSI per feature for a current matrix of shape (m, len(names)); returns shape (f,)."""
        current = np.asarray(current, dtype=np.float64)
        return np.array(
            [
                compute_psi(self.reference_counts[j], bin_counts(current[:, j], self.edges[j]), eps)
                for j in range(len(self.names))
            ]
        )


def psi_signals(
    profile: PsiProfile,
    current: np.ndarray,
    share_threshold: float = 0.2,
    eps: float = DEFAULT_FLOOR,
) -> dict[str, float]:
    """Return `psi_max` and `psi_share` (fraction of features with PSI above the threshold)."""
    values = profile.feature_psi(current, eps)
    return {"psi_max": float(values.max()), "psi_share": float((values > share_threshold).mean())}
