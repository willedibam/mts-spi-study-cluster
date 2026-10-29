"""One-knob generators for the constructed case studies (2026-10-28).

``generate_form_ladder_mts``: blocks of channels observe one shared latent, each block
through its own map g (linear, monotone or folded). Channel strength enters
*before* the map, ``x_i = g(w_i)`` with ``w_i = (u + s_i e_i) / sqrt(1 + s_i^2)``,
so a monotone g leaves every copula, hence Spearman rho and mutual information,
unchanged and moves only Pearson r.

``generate_drifting_delay_mts``: noisy copies of one smooth mother, each read at
its own delay plus a slow sinusoidal drift of that delay, ``x_i(t) = m(t - tau_i
- D c_i sin(2 pi t / P + theta_i))``. A constant delay (D = 0) is aligned by one
global shift; a drifting one only by a time-varying alignment.

``generate_delay_blocks_mts``: the same mother read by blocks of channels, each block
aligned, lagged (graded constant delays) or warped (graded time-varying delays about
zero). Lag and warp are switched on per block, so each misalignment can be studied
alone or together.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._common import _maybe_zscore, _resolve_rng

FORMS = ("linear", "monotone", "fold")
DELAYS = ("aligned", "lagged", "warped")


@dataclass
class FormLadderInternals:
    latent: np.ndarray       # (T,) standardised AR(1) latent u
    types: list              # per channel: its block's map
    noise_stds: np.ndarray   # (M,) latent noise s_i; loading a_i = 1 / sqrt(1 + s_i^2)


@dataclass
class DriftingDelayInternals:
    mother: np.ndarray       # fine-grid mother (OU process, `fine` points per sample)
    delays: np.ndarray       # (M,) constant delays tau_i, in samples
    drift_amp: np.ndarray    # (M,) drift amplitude D c_i, in samples
    drift_phase: np.ndarray  # (M,) theta_i
    noise_stds: np.ndarray   # (M,) observation noise


@dataclass
class DelayBlocksInternals:
    mother: np.ndarray       # fine-grid mother (OU process, `fine` points per sample)
    types: list              # per channel: its block's delay kind
    delay: np.ndarray        # (T, M) delay of channel i at time t, in samples
    lags: np.ndarray         # (M,) constant delay (lagged blocks), in samples
    warps: np.ndarray        # (M,) warp amplitude (warped blocks), in samples
    noise_stds: np.ndarray   # (M,) observation noise


def _ou_mother(T: int, pad: int, a1: float, fine: int, rng) -> np.ndarray:
    """Stationary OU process on a grid ``fine`` times finer than the samples, lag-one autocorrelation a1."""
    n = (T + 2 * pad) * fine
    a = a1 ** (1.0 / fine)
    m = np.empty(n)
    m[0] = rng.standard_normal()
    shocks = rng.standard_normal(n) * np.sqrt(1.0 - a * a)
    for k in range(1, n):
        m[k] = a * m[k - 1] + shocks[k]
    return m


def _ar1(T: int, a: float, rng, burn: int = 200) -> np.ndarray:
    e = rng.standard_normal(T + burn)
    m = np.zeros(T + burn)
    for t in range(1, T + burn):
        m[t] = a * m[t - 1] + e[t]
    m = m[burn:]
    return (m - m.mean()) / m.std()


def generate_form_ladder_mts(
    M: int = 12,
    T: int = 2000,
    maps: tuple = ("linear", "monotone", "fold"),
    curvature: float = 2.0,
    noise_low: float = 0.2,
    noise_high: float = 1.0,
    ar1_a: float = 0.8,
    zscore: bool = True,
    rng=None,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, FormLadderInternals]:
    """
    M channels in ``len(maps)`` equal blocks; block k observes the shared latent through ``maps[k]``.

    maps       : per block, "linear" (g = w), "monotone" (g = sinh(curvature w), odd, so its pairs with
                 folded channels have r = rho = 0 in population; curvature 0 is linear) or "fold" (g = w^2)
    noise_low, noise_high : latent noise s_i, log-spaced within each block (every block has the same
                 set, so strength spread does not depend on the maps)
    ar1_a      : AR(1) coefficient of the latent

    Returns shape (T, M), or ((T, M), FormLadderInternals).
    """
    maps = tuple(maps)
    if any(m not in FORMS for m in maps):
        raise ValueError(f"maps must be drawn from {FORMS} (got {maps!r}).")
    if M % len(maps) or M // len(maps) < 2:
        raise ValueError(f"M ({M}) must split into {len(maps)} equal blocks of at least 2 channels.")
    if not 0 < noise_low <= noise_high:
        raise ValueError(f"need 0 < noise_low <= noise_high (got {noise_low}, {noise_high}).")
    rng = _resolve_rng(None, rng)
    h = M // len(maps)
    u = _ar1(T, ar1_a, rng)
    s = np.tile(np.geomspace(noise_low, noise_high, h), len(maps))
    w = (u[:, None] + s * rng.standard_normal((T, M))) / np.sqrt(1.0 + s**2)
    x = w.copy()
    types = [m for m in maps for _ in range(h)]
    for k, form in enumerate(maps):
        block = slice(k * h, (k + 1) * h)
        if form == "monotone" and curvature != 0:
            x[:, block] = np.sinh(curvature * w[:, block])
        elif form == "fold":
            x[:, block] = w[:, block] ** 2
    data = _maybe_zscore(x, zscore=zscore)
    if return_internals:
        return data, FormLadderInternals(u, types, s)
    return data


def generate_drifting_delay_mts(
    M: int = 12,
    T: int = 1000,
    max_lag: float = 4.0,
    drift: float = 0.0,
    drift_period: float = 250.0,
    noise_low: float = 0.1,
    noise_high: float = 0.4,
    ar1_a: float = 0.9,
    fine: int = 8,
    structure_seed: int = 0,
    zscore: bool = True,
    rng=None,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, DriftingDelayInternals]:
    """
    Channel i reads one smooth mother at ``t - tau_i - drift c_i sin(2 pi t / drift_period + theta_i)``.

    max_lag    : constant delays tau_i = linspace(0, max_lag, M) (0 gives a synchronous system)
    drift      : drift amplitude D in samples; c_i ~ U(0, 1) so pairs drift by different amounts
    noise_low, noise_high : observation noise, log-spaced and assigned in a fixed shuffled order
    ar1_a      : lag-one autocorrelation of the mother at unit sampling. The mother is an
                 Ornstein-Uhlenbeck process on a grid ``fine`` times finer, read by linear
                 interpolation, so fractional and drifting delays do not smooth the signal.
    structure_seed : seed of c_i, theta_i and the noise order (shared by every replicate)

    Returns shape (T, M), or ((T, M), DriftingDelayInternals).
    """
    if max_lag < 0 or drift < 0:
        raise ValueError(f"max_lag and drift must be >= 0 (got {max_lag}, {drift}).")
    if not 0 < ar1_a < 1:
        raise ValueError(f"ar1_a must lie in (0, 1) (got {ar1_a}).")
    srng = np.random.default_rng(structure_seed)
    c = srng.uniform(0.0, 1.0, M)
    theta = srng.uniform(0.0, 2.0 * np.pi, M)
    noise = np.geomspace(noise_low, noise_high, M)[srng.permutation(M)]
    tau = np.linspace(0.0, float(max_lag), M)

    rng = _resolve_rng(None, rng)
    pad = int(np.ceil(max_lag + drift)) + 2
    n = (T + 2 * pad) * fine
    a = ar1_a ** (1.0 / fine)
    m = np.empty(n)
    m[0] = rng.standard_normal()
    shocks = rng.standard_normal(n) * np.sqrt(1.0 - a * a)
    for k in range(1, n):
        m[k] = a * m[k - 1] + shocks[k]
    t = np.arange(T, dtype=float)
    shift = tau[None] + drift * c[None] * np.sin(2.0 * np.pi * t[:, None] / drift_period + theta[None])
    grid = np.arange(n) / fine
    x = np.stack([np.interp(t + pad - shift[:, i], grid, m) for i in range(M)], axis=1)
    x = x + noise * rng.standard_normal((T, M))
    data = _maybe_zscore(x, zscore=zscore)
    if return_internals:
        return data, DriftingDelayInternals(m, tau, drift * c, theta, noise)
    return data


def generate_delay_blocks_mts(
    M: int = 15,
    T: int = 1000,
    blocks: tuple = ("aligned", "lagged", "warped"),
    lags: tuple = (1, 2, 3, 4, 5),
    warp: float = 6.0,
    warp_period: float = 250.0,
    noise_low: float = 0.1,
    noise_high: float = 0.4,
    noise_order: tuple | None = None,
    ar1_a: float = 0.9,
    fine: int = 8,
    zscore: bool = True,
    rng=None,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, DelayBlocksInternals]:
    """
    M channels in ``len(blocks)`` equal blocks of h read one smooth mother at ``t - delay_i(t)``.

    blocks     : per block, "aligned" (delay 0), "lagged" (channel k of the block delayed by ``lags[k]``)
                 or "warped" (``A_k sin(2 pi t / warp_period)``, zero mean, one phase, amplitudes
                 A_k = warp (k + 1) / h), so pairs differ in lag or in warp by graded amounts
    lags       : h constant delays in samples; whole samples are absorbed exactly by integer-shift
                 statistics such as xED, fractional ones are not
    noise_low, noise_high : observation noise, h log-spaced levels repeated in every block, assigned within
                 each block in ``noise_order`` (default ascending) so noise need not follow lag or warp
    ar1_a, fine : the mother is an OU process with lag-one autocorrelation ``ar1_a``, on a grid ``fine``
                 times finer and read by linear interpolation, so fractional delays do not smooth it

    Every random draw depends only on (M, T, lags, warp, ar1_a, fine) and the rng, never on ``blocks``,
    so a channel whose block kind is unchanged is identical across block layouts.

    Returns shape (T, M), or ((T, M), DelayBlocksInternals).
    """
    blocks = tuple(blocks)
    if any(b not in DELAYS for b in blocks):
        raise ValueError(f"blocks must be drawn from {DELAYS} (got {blocks!r}).")
    if M % len(blocks) or M // len(blocks) < 2:
        raise ValueError(f"M ({M}) must split into {len(blocks)} equal blocks of at least 2 channels.")
    h = M // len(blocks)
    lags = np.asarray(lags, float)
    order = np.arange(h) if noise_order is None else np.asarray(noise_order)
    if lags.shape != (h,) or np.any(lags < 0) or warp < 0:
        raise ValueError(f"need {h} non-negative lags and warp >= 0 (got {lags.tolist()}, {warp}).")
    if sorted(order.tolist()) != list(range(h)):
        raise ValueError(f"noise_order must be a permutation of range({h}) (got {order.tolist()}).")
    if not 0 < ar1_a < 1:
        raise ValueError(f"ar1_a must lie in (0, 1) (got {ar1_a}).")
    rng = _resolve_rng(None, rng)
    pad = int(np.ceil(lags.max() + warp)) + 2
    m = _ou_mother(T, pad, ar1_a, fine, rng)
    eps = rng.standard_normal((T, M))
    t = np.arange(T, dtype=float)
    lag_i, warp_i = np.zeros(M), np.zeros(M)
    for k, kind in enumerate(blocks):
        block = slice(k * h, (k + 1) * h)
        if kind == "lagged":
            lag_i[block] = lags
        elif kind == "warped":
            warp_i[block] = warp * np.arange(1, h + 1) / h
    delay = lag_i[None] + warp_i[None] * np.sin(2.0 * np.pi * t[:, None] / warp_period)
    noise = np.tile(np.geomspace(noise_low, noise_high, h)[order], len(blocks))
    grid = np.arange(m.size) / fine
    x = np.stack([np.interp(t + pad - delay[:, i], grid, m) for i in range(M)], axis=1) + noise * eps
    data = _maybe_zscore(x, zscore=zscore)
    if return_internals:
        return data, DelayBlocksInternals(m, [b for b in blocks for _ in range(h)], delay, lag_i, warp_i, noise)
    return data
