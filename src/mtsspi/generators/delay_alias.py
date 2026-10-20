"""Delayed and synchronous MTS with identical zero-lag correlation matrices.

Channel j is ``Y_j(t) = X_j(t - tau_j)``, where ``X = s @ L.T`` mixes M
independent unit-variance AR(1) sources with coefficient ``a``, so that
``Cov(X_j(t), X_k(t + h)) = S*_jk a^|h|`` with ``L @ L.T = S*``. The zero-lag
correlation of ``Y`` is ``S* o R`` with ``R_jk = a^|tau_j - tau_k|``. Mixing
with ``S* = S / R`` (entry-wise) therefore reproduces a target ``S`` exactly,
for every delay vector for which ``S / R`` is positive semidefinite.

For a generic target that holds only for small delay ranges. This generator
uses targets of the form ``S = S0 o R(tau_max)``, with a one-factor ``S0``
and integer channel positions ``tau_max``. Each delay vector
``tau = round_half_up(c * tau_max)``, ``0 <= c <= 1``, splits
``|tau_max_j - tau_max_k|`` into ``|tau_j - tau_k| + |w_j - w_k|`` with
``w = tau_max - tau``, because both ``tau`` and ``w`` are non-decreasing in
``tau_max``. Hence ``S / R(tau) = S0 o R(w)``, which is positive definite by
the Schur product theorem: every ``lag_range`` up to ``max_lag_range`` is
feasible and shares the same zero-lag matrix ``S``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._common import _maybe_zscore, _resolve_rng


@dataclass
class DelayAliasInternals:
    """Internals returned by generate_delay_aliased_mts when return_internals=True."""
    zero_lag_corr: np.ndarray   # (M, M) target S, shared by every lag_range
    intrinsic_corr: np.ndarray  # (M, M) S* = S / R(lags), mixes the sources
    lags: np.ndarray            # (M,) integer delay tau_j of each channel
    max_lags: np.ndarray        # (M,) integer channel positions tau_max
    loadings: np.ndarray        # (M,) one-factor loadings of S0


def ar1_lag_attenuation(lags: np.ndarray, ar1_a: float) -> np.ndarray:
    """R_jk = a^|tau_j - tau_k|: zero-lag attenuation of a pair by its delays."""
    lags = np.asarray(lags, dtype=float)
    return float(ar1_a) ** np.abs(lags[:, None] - lags[None, :])


def delay_alias_structure(
    M: int,
    max_lag_range: int = 15,
    ar1_a: float = 0.9,
    loading_low: float = 0.6,
    loading_high: float = 0.8,
    structure_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target zero-lag matrix S = S0 o R(tau_max), positions tau_max, loadings.

    Channel positions are ``round_half_up(linspace(0, max_lag_range, M))`` in
    channel order; loadings are ``Uniform(loading_low, loading_high)`` drawn
    from ``structure_seed`` only, so every family and replicate built with the
    same structure arguments shares ``S``.
    """
    if M < 2:
        raise ValueError(f"M must be >= 2 (got {M}).")
    if not 0.0 < ar1_a < 1.0:
        raise ValueError(f"ar1_a must lie in (0, 1) (got {ar1_a}).")
    if not 0.0 <= loading_low <= loading_high < 1.0:
        raise ValueError(
            f"need 0 <= loading_low <= loading_high < 1 (got {loading_low}, {loading_high})."
        )
    if int(max_lag_range) != max_lag_range or max_lag_range < 0:
        raise ValueError(f"max_lag_range must be a non-negative integer (got {max_lag_range}).")
    loadings = np.random.default_rng(structure_seed).uniform(loading_low, loading_high, M)
    base = np.outer(loadings, loadings)
    np.fill_diagonal(base, 1.0)
    max_lags = np.floor(np.linspace(0, int(max_lag_range), M) + 0.5).astype(int)
    return base * ar1_lag_attenuation(max_lags, ar1_a), max_lags, loadings


def generate_delay_aliased_mts(
    M: int,
    T: int,
    lag_range: float = 0.0,
    max_lag_range: int = 15,
    ar1_a: float = 0.9,
    loading_low: float = 0.6,
    loading_high: float = 0.8,
    structure_seed: int = 0,
    zscore: bool = True,
    rng=None,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, DelayAliasInternals]:
    """
    M-channel Gaussian MTS whose zero-lag correlation is the same matrix S for
    every lag_range; only the timing differs.

        lag_range = 0 : synchronous, tau_j = 0 and S* = S.
        lag_range > 0 : tau_j = round_half_up(lag_range / max_lag_range * tau_max_j),
                        so max_j tau_j - min_j tau_j = round_half_up(lag_range),
                        and S* = S / R(tau).

    Lag-tolerant similarities (maximum cross-correlation, coherence magnitude)
    of pair (j, k) estimate |S*_jk| = S_jk / a^|tau_j - tau_k|; zero-lag ones
    estimate S_jk in every family.

    Parameters
    ----------
    M, T           : channels and output length
    lag_range      : delay range in samples, 0 <= lag_range <= max_lag_range
    max_lag_range  : integer delay range at which S* = S0 (all structure from delays)
    ar1_a          : AR(1) coefficient of every source; correlation time -1/ln(a)
    loading_low, loading_high : range of the one-factor loadings of S0
    structure_seed : seed of the loadings; fixes S across families and replicates
    zscore         : z-score output
    rng            : RNG instance or int seed for the sources only
    return_internals : if True, return (data, DelayAliasInternals)

    Returns shape (T, M), or tuple of ((T, M), DelayAliasInternals).
    """
    target, max_lags, loadings = delay_alias_structure(
        M, max_lag_range, ar1_a, loading_low, loading_high, structure_seed
    )
    if not 0.0 <= lag_range <= max_lag_range:
        raise ValueError(f"need 0 <= lag_range <= max_lag_range (got {lag_range}, {max_lag_range}).")
    fraction = lag_range / max_lag_range if max_lag_range else 0.0
    lags = np.floor(fraction * max_lags + 0.5).astype(int)
    intrinsic = target / ar1_lag_attenuation(lags, ar1_a)
    # Positive definite by construction (module docstring); Cholesky enforces it.
    mixing = np.linalg.cholesky(intrinsic)

    rng = _resolve_rng(None, rng)
    n = T + int(lags.max())
    shocks = rng.standard_normal((n, M))
    shocks[1:] *= np.sqrt(1.0 - ar1_a**2)
    sources = np.empty_like(shocks)
    sources[0] = shocks[0]  # stationary start: unit variance at every t
    for t in range(1, n):
        sources[t] = ar1_a * sources[t - 1] + shocks[t]
    mixed = sources @ mixing.T

    start = int(lags.max()) - lags
    data = np.stack([mixed[start[j]: start[j] + T, j] for j in range(M)], axis=1)
    data = _maybe_zscore(data, zscore=zscore)
    if return_internals:
        return data, DelayAliasInternals(
            zero_lag_corr=target,
            intrinsic_corr=intrinsic,
            lags=lags,
            max_lags=max_lags,
            loadings=loadings,
        )
    return data
