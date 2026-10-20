"""Two-band MTS whose bands carry their own channel delays.

Channel j is ``Y_j(t) = sum_b [l_bj c_b(t - tau_bj) + sqrt(1 - l_bj^2) e_bj(t)] + n_j(t)``:
each band b has one common drive ``c_b`` and private processes ``e_bj`` with the
same raised-cosine spectrum ``s_b``, and ``n_j`` is white noise. With
non-overlapping bands the cross-spectrum of a pair is

    S_jk(f) = l_bj l_bk s_b(f) exp(-2 pi i f (tau_bj - tau_bk))   for f in band b,

so ``|S_jk(f)|`` and every auto-spectrum do not depend on the delays; only the
phase does. The three families share every spectral magnitude and differ in phase:

    synchronous : tau_bj = 0                       zero phase
    delayed     : tau_bj = tau_j for both bands    phase linear in f (one delay per pair)
    dispersive  : the fast band's delays are a     phase linear within each band with
                  fixed permutation of the slow    a different slope per band
                  band's

Synthesis is in the frequency domain on a length-T grid, so delays are exact
circular shifts and the sample has the population spectrum exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._common import _maybe_zscore, _resolve_rng

FAMILIES = ("synchronous", "delayed", "dispersive")


@dataclass
class BandDelayInternals:
    """Internals returned by generate_band_delayed_mts when return_internals=True."""
    delays: np.ndarray    # (B, M) integer delay of band b's common drive in channel j
    loadings: np.ndarray  # (B, M) loading l_bj of channel j on band b's common drive
    bands: tuple          # ((centre, half_width), ...) in cycles per sample
    band_var: np.ndarray  # (B,) variance of each band component
    noise_var: float      # white-noise variance per channel


def band_spectrum(f: np.ndarray, centre: float, half_width: float, var: float) -> np.ndarray:
    """Two-sided raised-cosine spectral density on |f| <= 1/2 integrating to ``var``.

    Support is ``||f| - centre| < half_width``; ``centre`` is 0 (low-pass) or at
    least ``half_width`` (band-pass), so the two halves never overlap.
    """
    u = (np.abs(np.asarray(f, dtype=float)) - centre) / half_width
    shape = np.where(np.abs(u) < 1.0, np.cos(0.5 * np.pi * u) ** 2, 0.0)
    return var * shape / (half_width if centre == 0 else 2.0 * half_width)


def band_delay_structure(
    M: int,
    family: str = "dispersive",
    lag_range: int = 15,
    n_bands: int = 2,
    loading_low: float = 0.5,
    loading_high: float = 0.9,
    shared_loadings: bool = False,
    structure_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Delays ``(B, M)`` and loadings ``(B, M)`` of one family.

    Positions are ``round_half_up(linspace(0, lag_range, M))`` in channel order.
    Loadings (per band, or one vector shared by all bands) and the dispersive
    permutation are drawn from ``structure_seed`` only, so every replicate with
    the same structure arguments shares them.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES} (got {family!r}).")
    if M < 2:
        raise ValueError(f"M must be >= 2 (got {M}).")
    if int(lag_range) != lag_range or lag_range < 0:
        raise ValueError(f"lag_range must be a non-negative integer (got {lag_range}).")
    if not 0.0 <= loading_low <= loading_high <= 1.0:
        raise ValueError(
            f"need 0 <= loading_low <= loading_high <= 1 (got {loading_low}, {loading_high})."
        )
    rng = np.random.default_rng(structure_seed)
    draws = rng.uniform(loading_low, loading_high, (n_bands, M))
    loadings = np.repeat(draws[:1], n_bands, axis=0) if shared_loadings else draws
    perms = [np.arange(M)] + [rng.permutation(M) for _ in range(n_bands - 1)]
    positions = np.floor(np.linspace(0, int(lag_range), M) + 0.5).astype(int)
    if family == "synchronous":
        delays = np.zeros((n_bands, M), dtype=int)
    elif family == "delayed":
        delays = np.repeat(positions[None], n_bands, axis=0)
    else:
        delays = np.stack([positions[p] for p in perms])
    return delays, loadings


def band_delay_cross_spectrum(internals: BandDelayInternals, f: np.ndarray) -> np.ndarray:
    """Population spectral matrix ``S[f, j, k]`` (two-sided density) at frequencies ``f``."""
    f = np.asarray(f, dtype=float)
    M = internals.delays.shape[1]
    diag = np.arange(M)
    S = np.zeros((len(f), M, M), dtype=complex)
    for (centre, half_width), var, lags, lam in zip(
        internals.bands, internals.band_var, internals.delays, internals.loadings
    ):
        s = band_spectrum(f, centre, half_width, var)
        a = lam[None, :] * np.exp(-2j * np.pi * f[:, None] * lags[None, :])
        S += s[:, None, None] * a[:, :, None] * a.conj()[:, None, :]
        S[:, diag, diag] += s[:, None] * (1.0 - lam**2)  # private part: full band power on the diagonal
    S[:, diag, diag] += internals.noise_var
    return S


def generate_band_delayed_mts(
    M: int,
    T: int,
    family: str = "dispersive",
    lag_range: int = 15,
    bands: tuple = ((0.0, 0.08), (0.25, 0.15)),
    band_var: tuple = (1.0, 1.0),
    noise_var: float = 0.5,
    loading_low: float = 0.5,
    loading_high: float = 0.9,
    shared_loadings: bool = False,
    structure_seed: int = 0,
    zscore: bool = True,
    rng=None,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, BandDelayInternals]:
    """
    M-channel Gaussian MTS with a slow and a fast band component, each with its
    own channel delays (module docstring). All families built with the same
    structure arguments share every auto-spectrum and every ``|S_jk(f)|``;
    they differ only in cross-spectral phase.

    Parameters
    ----------
    M, T            : channels and output length
    family          : "synchronous", "delayed" or "dispersive"
    lag_range       : integer delay range in samples (ignored for "synchronous")
    bands           : ((centre, half_width), ...) raised-cosine bands in cycles/sample;
                      they must not overlap
    band_var        : variance of each band component
    noise_var       : white-noise variance per channel
    loading_low, loading_high : range of the per-band loadings l_bj
    shared_loadings : one loading vector for all bands (coherence magnitude then
                      has the same pair pattern at every frequency)
    structure_seed  : seed of loadings and dispersive permutation
    zscore          : z-score output
    rng             : RNG instance or int seed for the processes only
    return_internals : if True, return (data, BandDelayInternals)

    Returns shape (T, M), or tuple of ((T, M), BandDelayInternals).
    """
    bands = tuple((float(c), float(w)) for c, w in bands)
    if len(band_var) != len(bands):
        raise ValueError("band_var needs one variance per band.")
    for c, w in bands:
        if w <= 0 or (c and c < w) or c + w > 0.5:
            raise ValueError(f"band {(c, w)} must have 0 < w, centre 0 or >= w, and centre + w <= 0.5.")
    edges = sorted((c - w if c else 0.0, c + w) for c, w in bands)
    if any(prev[1] > nxt[0] for prev, nxt in zip(edges, edges[1:])):
        raise ValueError(f"bands overlap: {bands}.")
    delays, loadings = band_delay_structure(
        M, family, lag_range, len(bands), loading_low, loading_high, shared_loadings, structure_seed
    )
    internals = BandDelayInternals(delays, loadings, bands, np.asarray(band_var, float), float(noise_var))

    rng = _resolve_rng(None, rng)
    f = np.fft.rfftfreq(T)
    X = np.zeros((len(f), M), dtype=complex)
    for (centre, half_width), var, lags, lam in zip(bands, band_var, delays, loadings):
        amp = np.sqrt(band_spectrum(f, centre, half_width, var))[:, None]
        common = np.fft.rfft(rng.standard_normal(T))[:, None]
        private = np.fft.rfft(rng.standard_normal((T, M)), axis=0)
        X += amp * (lam * common * np.exp(-2j * np.pi * f[:, None] * lags[None, :])
                    + np.sqrt(1.0 - lam**2) * private)
    data = np.fft.irfft(X, n=T, axis=0) + np.sqrt(noise_var) * rng.standard_normal((T, M))
    data = _maybe_zscore(data, zscore=zscore)
    if return_internals:
        return data, internals
    return data
