"""
Node feature extraction from raw multivariate time series.

Simple univariate summaries per channel — no temporal encoder.
Keeps the first experiments interpretable: if the model works,
it's because of edge attributes, not a powerful node encoder.
"""
from __future__ import annotations

import numpy as np


def node_features(mts: np.ndarray) -> np.ndarray:
    """
    Compute per-channel summary features from a (T, M) time series.

    Features per node (channel):
        0: mean
        1: std
        2: lag-1 autocorrelation
        3: dominant FFT magnitude (excluding DC)

    Returns shape (M, 4).
    """
    mts = np.asarray(mts, dtype=np.float64)
    if mts.ndim != 2:
        raise ValueError(f"Expected (T, M) array, got shape {mts.shape}")
    T, M = mts.shape

    feats = np.zeros((M, 4), dtype=np.float32)

    for i in range(M):
        x = mts[:, i]

        # Mean, std
        feats[i, 0] = x.mean()
        s = x.std()
        feats[i, 1] = s

        # Lag-1 autocorrelation
        if T > 1 and s > 1e-12:
            x_centered = x - x.mean()
            c0 = np.dot(x_centered, x_centered)
            c1 = np.dot(x_centered[:-1], x_centered[1:])
            feats[i, 2] = c1 / c0 if c0 > 1e-12 else 0.0
        else:
            feats[i, 2] = 0.0

        # Dominant FFT magnitude (excluding DC)
        if T > 1:
            fft_mag = np.abs(np.fft.rfft(x))
            fft_mag[0] = 0.0  # exclude DC
            feats[i, 3] = fft_mag.max()
        else:
            feats[i, 3] = 0.0

    return feats
