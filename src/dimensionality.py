from __future__ import annotations

import numpy as np
from scipy.stats import zscore


def explained_variance_curve(data: np.ndarray) -> np.ndarray:
    """Cumulative explained variance ratio across principal components of an MTS.

    Parameters
    ----------
    data : np.ndarray
        MTS of shape (T, M): T timesteps, M channels.

    Returns
    -------
    np.ndarray
        Cumulative explained variance ratios, length min(T, M).

    Notes
    -----
    Channels are z-scored before SVD (correlation-matrix PCA). This matches
    what most SPIs see (scale-invariant or internally z-scored) and avoids
    (a) scale-driven PC ordering and (b) catastrophic rank collapse on
    heavy-tailed channels whose raw variance is dominated by a few outliers.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (T, M); got shape {data.shape}")
    X = zscore(data, axis=0, nan_policy="omit")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    var = s ** 2
    total = var.sum()
    if total <= 0:
        return np.zeros_like(var)
    return np.cumsum(var) / total


def n_pcs_for_variance(data: np.ndarray, alpha: float = 0.95) -> int:
    """Smallest number of PCs whose cumulative explained variance ≥ alpha.

    Returns an integer in [1, min(T, M)]. Constant-variance or all-zero data
    (after z-scoring) returns min(T, M) as a conservative fallback; this is
    the only branch where the result is not strictly interpretable as
    intrinsic dimensionality.
    """
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
    cum = explained_variance_curve(data)
    if cum.size == 0:
        return 0
    if cum[-1] < alpha - 1e-12:
        return int(cum.size)
    return int(np.searchsorted(cum, alpha) + 1)
