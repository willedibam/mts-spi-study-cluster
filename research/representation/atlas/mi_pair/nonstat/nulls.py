"""Non-stationary linear nulls for the MI pair, applied to a normal-scored (T, M) window v.

N0  joint IAAFT, spectrum-final (the anchor's null): stationary linear-Gaussian process with v's auto- and cross-spectra.
N1  local-gain null: v = a(t) * u(t) per channel, a = Gaussian-smoothed Hilbert envelope (sigma samples), u = v / a; joint IAAFT of u,
    remodulated by the same a, then normal-scored as v is. H0: a stationary linear network under each channel's own slow gain.
    Keeping every channel's envelope also keeps envelope co-fluctuation, so it is generous to the null.
    NOT USED: miscalibrated in the pilot (calib.py docstring). With a floor that the gain does not scale, u keeps a gain-dependent
    correlation structure that IAAFT makes stationary, and the surrogate gains non-Gaussian dependence the data lack.
N2  segment null: joint IAAFT, spectrum-final, on consecutive segments of L samples, concatenated. H0: linear-Gaussian dependence
    that may change between segments (gain with any noise floor, slow coupling switching, slow spectral drift).
Background: stationary surrogates read non-stationarity as nonlinearity (Schreiber and Schmitz 2000; Lancaster et al. 2018).
"""
import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, rankdata
from mtsspi.surrogates import multivariate_iaaft

ns = lambda x: norm.ppf((np.apply_along_axis(rankdata, 0, x) - .5) / len(x))


def n0(v, rng):
    return multivariate_iaaft(v, rng, final='spectrum')[0]


def envelope(v, sigma):
    a = gaussian_filter1d(np.abs(hilbert(v, axis=0)), sigma, axis=0, mode='reflect')
    a = a / a.mean(0); return np.maximum(a, .05)


def n1(v, rng, sigma):
    a = envelope(v, sigma); u = multivariate_iaaft(v / a, rng, final='spectrum')[0]
    return ns(a * u)


def n2(v, rng, L):
    T = len(v); cuts = list(range(0, T, L)) + [T]
    if len(cuts) > 2 and cuts[-1] - cuts[-2] < L // 2: del cuts[-2]   # fold a short tail into the previous segment
    out = np.empty_like(v)
    for s, e in zip(cuts[:-1], cuts[1:]): out[s:e] = multivariate_iaaft(v[s:e], rng, final='spectrum')[0]
    return out
