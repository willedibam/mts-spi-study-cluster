"""Label-free feature blocks added to existing banks for the comparator ladder.

Every block is a function of the stored bank or its raw windows alone. Only the
tangent-space reference is fitted, on the training rows it is given.
"""
import warnings

import numpy as np
from sklearn.covariance import ledoit_wolf


def mean_spi(aligned, spi_count):
    """K edge means: each SPI matrix averaged over its off-diagonal entries (strength in z's invariance class)."""
    profiles = np.asarray(aligned, dtype=np.float64).reshape(len(aligned), spi_count, -1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(profiles, axis=2)


def log_spectra(x, bands=32):
    """Per-channel log power in 32 equal-width bands, as dataset_screen.prepare.univariate; x is (n, T, M)."""
    centered = x - x.mean(1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)[:, 1:]) ** 2 / x.shape[1]
    means = np.stack([part.mean(1) for part in np.array_split(power, bands, axis=1)], axis=1)
    return np.log(np.maximum(means, np.finfo(float).tiny)).reshape(len(x), -1)


def catch22(x):
    """catch22 of every channel; x is (n, T, M), output (n, 22 M)."""
    import pycatch22
    return np.array([[v for m in range(x.shape[2]) for v in pycatch22.catch22_all(row[:, m].tolist())['values']]
                     for row in np.asarray(x, dtype=np.float64)])


def _power(matrix, exponent):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.maximum(values, 1e-12) ** exponent) @ vectors.T


def _log(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.log(np.maximum(values, 1e-12))) @ vectors.T


def tangent(x, train, iterations=50):
    """Tangent-space covariance: Ledoit-Wolf covariance per window, projected at the geometric mean of the training
    windows (Barachant et al. 2012; Varoquaux et al. 2010). x is (n, T, M); output (n, M (M + 1) / 2)."""
    covariances = np.array([ledoit_wolf(np.asarray(row, dtype=np.float64))[0] for row in x])
    reference = covariances[train].mean(0)
    for _ in range(iterations):
        root, inverse = _power(reference, .5), _power(reference, -.5)
        step = np.mean([_log(inverse @ c @ inverse) for c in covariances[train]], axis=0)
        values, vectors = np.linalg.eigh(step)
        reference = root @ ((vectors * np.exp(values)) @ vectors.T) @ root
        if np.linalg.norm(step) < 1e-10:
            break
    inverse = _power(reference, -.5)
    upper = np.triu_indices(covariances.shape[1])
    weight = np.where(upper[0] == upper[1], 1., np.sqrt(2.))
    return np.array([_log(inverse @ c @ inverse)[upper] * weight for c in covariances])
