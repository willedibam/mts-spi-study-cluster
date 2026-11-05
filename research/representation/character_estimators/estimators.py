"""Per-edge character estimators: four strength-free axes on [0, 1] for one pair of channels.

One principle: an axis is the share of the pair's dependence that needs one named property, defined on the
process first and estimated second. Every axis is unchanged by rescaling the coupling or the noise (at
population level), is computed only where the dependence it is a share of exceeds an independence null, and
is invariant to a static monotone distortion of either channel, because both channels are first mapped to
normal scores (``gaussianise``). README.md gives the definitions and their test systems.

- nonlinearity N (directional, x -> y; y -> x reported as ``nonlinearity_yx``): 1 - (variance of y predictable
  linearly from a window of x) / (variance predictable by a second-order Volterra model with cubic diagonal),
  cross-validated and corrected for each model's out-of-sample cost (measured on the circular shifts). Linearity
  of a regression is not symmetric outside the Gaussian case (on/off linear coupling makes E[x | y] nonlinear),
  so each direction has its own value; edges are ordered anyway.
- lag share L: share of the cross-spectral coupling power in quadrature, sum_f Im(C)^2 / sum_f |C|^2.
- variability V: share of the local coupling power that the recording's average coupling does not reproduce,
  1 - sum_f |E c|^2 / sum_f E|c|^2, with c a W-sample segment's power-normalised cross-spectrum after removing
  the fixed delay. Change in strength (V = 1 - q for coupling present in a fraction q of segments) and in timing
  (V = 1 - |E exp(i phase)|^2 for phase drift between segments) both count. Common amplitude modulation of both
  channels does not: it changes power, not coupling. Splitting V into strength and timing needs fourth moments
  of 25-sample estimates, which are unstable at T = 2000 (README, v4).
- directionality D (auxiliary: a normalised form of time-reversed Granger causality, which already exists): |(G_xy - G_yx) - (G~_xy - G~_yx)| / (sum of the four), G linear Granger, ~ time reversed,
  so asymmetry that time reversal does not flip (unequal noise, instantaneous mixing) does not count.

Estimates are not clipped to [0, 1]; ``delay`` (samples) is reported as an auxiliary coordinate.
The spectral axes (L, V) are computed on zero-phase whitened channels. Squared quantities are estimated
without noise bias from products of estimates on time-disjoint data (distinct segments, or distinct quarters of
a segment); products over the same samples would carry fourth-order cumulants for non-Gaussian signals.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import signal, stats

W = 100          # segment length in samples: the timescale of variability
NW, TAPERS = 2.5, 4
P_WIN = 3        # Volterra window half-width in lags
TAU_MAX = 10     # search range for the Volterra window centre
ALIGN_MAX = 20   # search range for the fixed delay removed before the local (quarter) analysis
AR_ORDER = 4     # Granger order
FOLDS = 5
SHIFTS = 39      # circular-shift independence null; p = (1 + #null >= obs) / (1 + SHIFTS)
ALPHA = .05
AXES = ('nonlinearity', 'lag', 'variability', 'direction', 'nonlinearity_raw')


def gaussianise(v):
    """Normal scores: removes any static monotone distortion of the channel."""
    return stats.norm.ppf((stats.rankdata(v) - .5) / len(v))


def standardise(v):
    return (v - v.mean()) / v.std()


def circular(v):
    """Remove the line joining the end points, so circular shifts and FFT whitening add no jump. Used only on
    the spectral path: the ramp would distort the polynomial terms of the nonlinearity regression."""
    v = np.asarray(v, float)
    return v - (v[-1] - v[0]) * np.arange(len(v)) / (len(v) - 1)


def whiten(v, nper=256):
    """Zero-phase spectral whitening. A real positive filter on each channel leaves coherency unchanged, and a
    flat spectrum makes the orthogonal-taper estimates independent, which the unbiased products need."""
    f, p = signal.welch(v, nperseg=min(nper, len(v) // 4))
    amp = np.sqrt(np.interp(np.fft.rfftfreq(len(v)), f, p))
    return np.fft.irfft(np.fft.rfft(v) / amp, n=len(v))


# ---- spectral axes -------------------------------------------------------------------------------------

def _tapered(x, y, w, pieces=1):
    """Multitaper eigencoefficients of each of ``pieces`` equal parts of each W-sample segment:
    (segments, pieces, tapers, frequencies)."""
    k, h = len(x) // w, w // pieces
    xs, ys = (v[:k * w].reshape(k, pieces, h) for v in (x, y))
    xs, ys = xs - xs.mean(-1, keepdims=True), ys - ys.mean(-1, keepdims=True)
    tap = signal.windows.dpss(h, NW, TAPERS)
    X = np.fft.rfft(xs[:, :, None] * tap, axis=-1)[..., 1:(h + 1) // 2]  # no DC, no Nyquist
    Y = np.fft.rfft(ys[:, :, None] * tap, axis=-1)[..., 1:(h + 1) // 2]
    return X, Y


def _coherency(X, Y, local=False):
    """Taper-averaged cross-spectrum in global coherency units: (segments, pieces, frequencies). With ``local``,
    each piece is also divided by its own broadband power in each channel, so common amplitude modulation of
    the channels cancels and only the coupling's own change remains; each piece's value depends only on its own
    samples, so products of disjoint pieces stay unbiased."""
    ax, ay = np.abs(X) ** 2, np.abs(Y) ** 2
    Sxx, Syy = ax.mean((0, 1, 2)), ay.mean((0, 1, 2))
    c = (X * Y.conj()).mean(2) / np.sqrt(Sxx * Syy)
    if local:
        c = c / np.sqrt((ax / Sxx).mean((2, 3)) * (ay / Syy).mean((2, 3)))[..., None]
    return c


def _distinct(c):
    """sum_f |E c(f)|^2 over segments, from products of distinct (time-disjoint) segments."""
    k = len(c)
    return (np.abs(c.sum(0)) ** 2 - (np.abs(c) ** 2).sum(0)) / (k * (k - 1))


def _lag(c):
    """Lag share and the debiased summed squared coherence it is a share of (c: segments x frequencies)."""
    k = len(c)
    c2 = _distinct(c)
    i2 = (c.imag.sum(0) ** 2 - (c.imag ** 2).sum(0)) / (k * (k - 1))
    return i2.sum() / c2.sum(), c2.sum()


QUARTERS = tuple(combinations(range(4), 2))


def _local(c):
    """Variability and mean local coupling power (c: segments x 4 quarters x frequencies, each quarter
    normalised by its own power).

    Products are taken only between time-disjoint quarters or segments, so they stay unbiased for
    non-Gaussian signals (products over the same samples carry fourth-order cumulants). L2 = E sum_f |c_k|^2,
    from products of distinct quarters of a segment; G = sum_f |E c|^2, from products of distinct segments.
    variability = 1 - G / L2. L2 pairs quarters 25 to 75 samples apart, so it is the coupling power at that
    resolution: change faster than that reads partly as weaker coupling, not as variability.
    """
    L2 = np.mean([(c[:, q] * c[:, r].conj()).real.sum(-1) for q, r in QUARTERS])
    G = _distinct(c.mean(1)).sum()
    return (1 - G / L2 if L2 > 0 else np.nan), L2


def _delay(xw, yw):
    """Fixed delay of y behind x (samples): the peak of |cross-correlation| within ALIGN_MAX."""
    r = np.fft.irfft(np.fft.rfft(xw).conj() * np.fft.rfft(yw), n=len(xw))
    lags = np.r_[0:ALIGN_MAX + 1, -ALIGN_MAX:0]
    return int(lags[np.argmax(np.abs(r[lags]))])


def _spectral(xw, yw, w):
    """Lag share on the channels as recorded; variability after removing the fixed delay,
    since a 25-sample quarter cannot resolve coupling at a delay of more than a few samples and a fixed delay is
    not a change over time."""
    lag, coh = _lag(_coherency(*_tapered(xw, yw, w))[:, 0])
    d = _delay(xw, yw)
    var, local = _local(_coherency(*_tapered(xw, np.roll(yw, -d), w, 4), local=True))
    return lag, coh, var, local, d


# ---- nonlinearity --------------------------------------------------------------------------------------

def _he2(v):
    return (v ** 2 - 1) / np.sqrt(2)


def _window(src, lags):
    """Rows t where src[t - l] exists for every lag l, and the lagged columns."""
    n, lo, hi = len(src), min(lags), max(lags)
    t = np.arange(max(0, hi), min(n, n + lo))
    return t, np.column_stack([src[t - l] for l in lags])


def _centre(src, dst):
    best, score = 0, -1.
    for tau in range(-TAU_MAX, TAU_MAX + 1):
        t, a = _window(src, [tau])
        a, b = a[:, 0], dst[t]
        s = np.corrcoef(a, b)[0, 1] ** 2 + np.corrcoef(_he2(a), b)[0, 1] ** 2 + np.corrcoef(a, _he2(b))[0, 1] ** 2
        if s > score:
            best, score = tau, s
    return best


def _cv_r2(A, b):
    A = np.column_stack([np.ones(len(b)), A])
    sse = sst = 0.
    for te in np.array_split(np.arange(len(b)), FOLDS):
        tr = np.ones(len(b), bool); tr[te] = False
        coef = np.linalg.lstsq(A[tr], b[tr], rcond=None)[0]
        sse += ((b[te] - A[te] @ coef) ** 2).sum(); sst += ((b[te] - b[tr].mean()) ** 2).sum()
    return 1 - sse / sst


def _volterra(src, dst):
    """Cross-validated R^2 of the linear window and of the full model, predicting dst from src."""
    c = _centre(src, dst)
    t, lin = _window(src, list(range(c - P_WIN, c + P_WIN + 1)))
    quad = np.column_stack([lin[:, i] * lin[:, j] for i, j in combinations(range(lin.shape[1]), 2)] + [lin ** 2, lin ** 3])
    return _cv_r2(lin, dst[t]), _cv_r2(np.column_stack([lin, quad]), dst[t])


def _fits(x, y):
    """(lin x->y, full x->y, lin y->x, full y->x)."""
    return np.array(_volterra(x, y) + _volterra(y, x))


def _nonlinearity(fits, null_mean):
    """Nonlinear share of predicting y from x and of predicting x from y, from out-of-sample R^2 corrected for each model's
    out-of-sample cost. Under independence a model's cross-validated R^2 is -k, its cost (about 0.004 for the
    linear and 0.03-0.05 for the full model at T = 2000), and in general R^2_cv = 1 - (1 - R^2)(1 + k); so
    R^2 = 1 - (1 - R^2_cv) / (1 - R^2_null). Without the correction, weak dependence makes the full model's R^2
    fall below the linear one's and the share goes strongly negative. ``null_mean`` is the mean of ``fits`` over
    the circular shifts, which carries the pair's own autocorrelation and marginals."""
    r = 1 - (1 - fits) / (1 - null_mean)
    return tuple(1 - max(lin, 0) / full if full > 0 else np.nan for lin, full in ((r[0], r[1]), (r[2], r[3])))


def _analytic_null(n):
    """Expected null cross-validated R^2 of each model for white regressors (used without the shift null)."""
    ntr, kl, kf = n * (FOLDS - 1) / FOLDS, 2 * P_WIN + 1, 2 * P_WIN + 1 + (2 * P_WIN + 1) * (2 * P_WIN + 4) // 2
    return -np.array([kl / (ntr - kl - 1), kf / (ntr - kf - 1)] * 2)


# ---- directionality ------------------------------------------------------------------------------------

def _granger(src, dst, p=AR_ORDER):
    n = len(dst)
    own = np.column_stack([dst[p - k:n - k] for k in range(1, p + 1)])
    cross = np.column_stack([src[p - k:n - k] for k in range(1, p + 1)])
    b, one = dst[p:], np.ones((n - p, 1))
    rss = [np.sum((b - A @ np.linalg.lstsq(A, b, rcond=None)[0]) ** 2)
           for A in (np.hstack([one, own]), np.hstack([one, own, cross]))]
    return np.log(rss[0] / rss[1]) - p / (n - 3 * p - 1)  # minus its asymptotic null mean


def _direction(x, y):
    g = np.clip([_granger(x, y), _granger(y, x), _granger(x[::-1], y[::-1]), _granger(y[::-1], x[::-1])], 0, None)
    d = abs((g[0] - g[1]) - (g[2] - g[3])) / g.sum() if g.sum() > 0 else np.nan
    return d, g[0] + g[1]


def _direction_floor(x, y):
    return _granger(x, y) + _granger(y, x)


# ---- all axes ------------------------------------------------------------------------------------------

def _value(v):
    """Per-edge estimates are not clipped: the population values lie in [0, 1], and clipping noisy estimates
    would bias every average of them upward."""
    return float(v) if np.isfinite(v) else np.nan


def estimate(x, y, rng=None, floor=True, w=W, shifts=SHIFTS):
    """All axes for one pair. ``floor`` tests each axis's dependence against ``shifts`` circular shifts of y."""
    xr, yr = standardise(x), standardise(y)
    xg, yg = gaussianise(x), gaussianise(y)
    xw, ycw = whiten(circular(xg)), circular(yg)  # circular only on the spectral path (FFT whitening, shifts)
    lag, coh, var, local, delay = _spectral(xw, whiten(ycw), w)
    fg, fr = _fits(xg, yg), _fits(xr, yr)
    d, gsum = _direction(xg, yg)
    out = dict(lag=lag, variability=var, direction=d)
    out.update(delay=delay, pearson=float(np.corrcoef(xr, yr)[0, 1]), coherence=float(coh / (w // 2 - 1)),
               local_coupling=float(local / ((w // 4 + 1) // 2 - 1)))
    null_g = null_r = _analytic_null(len(x))
    if floor:
        obs = dict(nonlinearity=fg[1], nonlinearity_raw=fr[1], nonlinearity_yx=fg[3], nonlinearity_raw_yx=fr[3],
                   lag=coh, variability=local, direction=gsum)
        null = {k: [] for k in obs}
        fits_g, fits_r = [], []
        rng = rng or np.random.default_rng(0)
        n = len(x)
        for s in rng.integers(w, n - w, shifts):
            ys = np.roll(yg, s)
            _, cs, _, loc, _ = _spectral(xw, whiten(np.roll(ycw, s)), w)
            null['lag'].append(cs); null['variability'].append(loc)
            fits_g.append(_fits(xg, ys)); fits_r.append(_fits(xr, np.roll(yr, s)))
            for k, f in (('', fits_g[-1]), ('_raw', fits_r[-1])):
                null[f'nonlinearity{k}'].append(f[1]); null[f'nonlinearity{k}_yx'].append(f[3])
            null['direction'].append(_direction_floor(xg, ys))
        null_g, null_r = np.mean(fits_g, 0), np.mean(fits_r, 0)
    out['nonlinearity'], out['nonlinearity_yx'] = _nonlinearity(fg, null_g)
    out['nonlinearity_raw'], out['nonlinearity_raw_yx'] = _nonlinearity(fr, null_r)
    for k in AXES + ('nonlinearity_yx', 'nonlinearity_raw_yx'):
        out[k] = _value(out[k])
    if floor:
        for k in obs:
            p = (1 + np.sum(np.array(null[k]) >= obs[k])) / (1 + shifts)
            out[f'{k}_defined'] = bool(p <= ALPHA)
            if not out[f'{k}_defined']:
                out[k] = np.nan
    return out
