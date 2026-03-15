from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.random import default_rng

GeneratorFn = Callable[..., np.ndarray]


def _global_rng():
    return default_rng(123456789)


def _resolve_rng(seed: int | None, rng=None):
    if rng is not None:
        if isinstance(rng, (int, np.integer)):
            return default_rng(int(rng))
        return rng
    if seed is None:
        return _global_rng()
    return default_rng(seed)


def _zscore_channels(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, ddof=0, keepdims=True)
    zero_mask = std < 1e-12
    std = np.where(zero_mask, 1.0, std)
    normalised = (data - mean) / std
    if np.any(zero_mask):
        cols = zero_mask.reshape(-1)
        eps = 1e-6
        normalised[0, cols] = eps
        if normalised.shape[0] > 1:
            normalised[1, cols] = -eps
    return normalised


def _maybe_zscore(data: np.ndarray, *, zscore: bool = True) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    return _zscore_channels(arr) if zscore else arr


def _resolve_channel_noise_stds(
    noise_std: float,
    M: int,
    *,
    noise_std_variable: bool = False,
    noise_std_scale: float = np.e,
    rng,
) -> np.ndarray:
    """Return per-channel noise std array of shape (M,).

    noise_std_variable=False: all channels use noise_std (no change).
    noise_std_variable=True:  channel i uses noise_std * kappa_i,
        where kappa_i ~ Geom(p=1/noise_std_scale) i.i.d.
        E[kappa] = noise_std_scale, so expected noise std = noise_std * noise_std_scale.
    """
    if not noise_std_variable:
        return np.full(M, float(noise_std))
    p = 1.0 / float(noise_std_scale)
    kappa = rng.geometric(p, size=M).astype(float)
    return float(noise_std) * kappa
