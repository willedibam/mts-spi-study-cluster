"""Two focused controls: cross-SPI dyad alignment and direct VAR(1) estimates."""
import numpy as np


def permute_dyads(mpis, order, rng, *, shared=False):
    """Move unordered dyads, optionally swapping their two directions together.

    Preserves each SPI's edge multiset, symmetry, and reciprocal-edge joint
    distribution. Independent maps remove cross-SPI dyad correspondence. A
    shared map is a common ordered-edge permutation and must preserve z.
    These perturbed MPI collections need not be realizable by a time series.
    """
    m = len(mpis[order[0]])
    upper = np.triu_indices(m, 1)
    lower = upper[::-1]
    count = len(upper[0])
    common = None
    result = {}
    for name in order:
        matrix = np.asarray(mpis[name])
        if matrix.shape != (m, m):
            raise ValueError("inconsistent MPI shape")
        if common is None or not shared:
            permutation = rng.permutation(count)
            flip = rng.integers(2, size=count).astype(bool)
            common = permutation, flip
        permutation, flip = common
        a, b = matrix[upper][permutation], matrix[lower][permutation]
        moved = matrix.copy()
        moved[upper] = np.where(flip, b, a)
        moved[lower] = np.where(flip, a, b)
        result[name] = moved
    return result


def linear_dynamics_features(raw, ridge_fraction=1e-3):
    """Estimate mean self-memory and mean signed total cross-channel coefficient.

    Fit centered Y=X B by ridge with lambda=1e-3*trace(X'X)/M. This fixed,
    amplitude-scaled stabilization uses no class labels. B is the transpose of
    the column-state transition matrix. No topology or true coefficient is used.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or len(raw) < 3 or raw.shape[1] < 2 or not np.isfinite(raw).all():
        raise ValueError("expected finite T by M data")
    if ridge_fraction <= 0:
        raise ValueError("ridge fraction must be positive")
    x, y = raw[:-1].copy(), raw[1:].copy()
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    gram = x.T @ x
    m = raw.shape[1]
    penalty = ridge_fraction * np.trace(gram) / m
    if penalty == 0:
        return np.full(2, np.nan)
    coefficients = np.linalg.solve(gram + penalty * np.eye(m), x.T @ y)
    return np.asarray([np.trace(coefficients) / m,
                       (coefficients.sum() - np.trace(coefficients)) / m])
