"""Per-edge character estimators: five strength-free axes on [0, 1] for one pair of channels.

One principle: an axis is the share of the pair's dependence that needs one named property, defined on the
process first and estimated second. Every axis is unchanged by rescaling the coupling or the noise (at
population level), is computed only where the dependence it is a share of exceeds an independence null, and
is invariant to a static monotone distortion of either channel, because both channels are first mapped to
normal scores (``gaussianise``). README.md gives the definitions and their test systems.

- nonlinearity N: 1 - (variance of one channel predictable linearly from a window of the other) / (variance
  predictable by a second-order Volterra model with cubic diagonal), cross-validated, in the better-predicted
  direction, after unit-variance scaling within each segment (slow amplitude modulation belongs to S).
- lag share L: share of the cross-spectral coupling power in quadrature, sum_f Im(C)^2 / sum_f |C|^2.
- lag variability V: 1 - (sum_f |E c|^2 / sum_f E|c|^2) / ((E P)^2 / E P^2), with c a W-sample segment's
  cross-spectrum and P = sum_f |c|^2 its coupling power; V = 1 - E cos(phase drift between segments) when the
  gain is constant or on/off.
- intermittency S: 1 - (E P)^2 / E P^2; S = 1 - q for coupling present in a fraction q of segments.
- directionality D: |(G_xy - G_yx) - (G~_xy - G~_yx)| / (sum of the four), G linear Granger, ~ time reversed,
  so asymmetry that time reversal does not flip (unequal noise, instantaneous mixing) does not count.

The spectral axes (L, V, S) are computed on zero-phase whitened channels. Squared quantities are estimated
without noise bias from products of estimates on time-disjoint data (distinct segments, or distinct quarters of
a segment); products over the same samples would carry fourth-order cumulants for non-Gaussian signals.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import signal, stats

W = 100          # segment length in samples: the timescale of lag variability and intermittency
NW, TAPERS = 2.5, 4
P_WIN = 3        # Volterra window half-width in lags
TAU_MAX = 10     # search range for the Volterra window centre
AR_ORDER = 4     # Granger order
FOLDS = 5
SHIFTS = 39      # circular-shift independence null; p = (1 + #null >= obs) / (1 + SHIFTS)
ALPHA = .05
AXES = ('nonlinearity', 'lag', 'lag_variability', 'intermittency', 'direction', 'nonlinearity_raw')


def gaussianise(v):
    """Normal scores: removes any static monotone distortion of the channel."""
    return stats.norm.ppf((stats.rankdata(v) - .5) / len(v))


def standardise(v):
    return (v - v.mean()) / v.std()


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


def _coherency(X, Y):
    """Taper-averaged cross-spectrum in global coherency units: (segments, pieces, frequencies)."""
    Sxx, Syy = (np.abs(X) ** 2).mean((0, 1, 2)), (np.abs(Y) ** 2).mean((0, 1, 2))
    return (X * Y.conj()).mean(2) / np.sqrt(Sxx * Syy)


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
MATCHINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def _local(c):
    """Lag variability, intermittency and mean local coupling power (c: segments x 4 quarters x frequencies).

    Products are taken only between time-disjoint quarters or segments, so they stay unbiased for
    non-Gaussian signals (products over the same samples carry fourth-order cumulants). With P_k(q, r) =
    sum_f Re c_kq c_kr*, and expectations over segments: L2 = E P = E sum_f |c_k|^2, L4 = E P(q, r) P(s, u)
    over disjoint quarter pairs = E (sum_f |c_k|^2)^2, L22 = (E P)^2 from distinct segments, G = sum_f |E c|^2
    from distinct segments. intermittency = 1 - L22 / L4; lag variability = 1 - (G / L2) / (L22 / L4).
    For a gain that is constant or on/off and common to all frequencies, lag variability =
    1 - E cos(phase_j - phase_k); a continuously varying gain makes it read low, not high.
    """
    P = {qr: (c[:, qr[0]] * c[:, qr[1]].conj()).real.sum(-1) for qr in QUARTERS}
    g = np.mean(list(P.values()), 0)
    k = len(g)
    L2 = g.mean()
    L4 = np.mean([P[a] * P[b] for a, b in MATCHINGS])
    L22 = (g.sum() ** 2 - (g ** 2).sum()) / (k * (k - 1))
    G = _distinct(c.mean(1)).sum()
    if L2 <= 0 or L4 <= 0 or L22 <= 0:
        return np.nan, np.nan, L2
    return 1 - (G / L2) / (L22 / L4), 1 - L22 / L4, L2


def _spectral(xw, yw, w):
    lag, coh = _lag(_coherency(*_tapered(xw, yw, w))[:, 0])
    stab, stead, local = _local(_coherency(*_tapered(xw, yw, w, 4)))
    return lag, coh, stab, stead, local


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


def _volterra(src, dst, full_only=False):
    c = _centre(src, dst)
    t, lin = _window(src, list(range(c - P_WIN, c + P_WIN + 1)))
    quad = np.column_stack([lin[:, i] * lin[:, j] for i, j in combinations(range(lin.shape[1]), 2)] + [lin ** 2, lin ** 3])
    full = _cv_r2(np.column_stack([lin, quad]), dst[t])
    return full if full_only else (_cv_r2(lin, dst[t]), full)


def local_standardise(v, w=W):
    """Unit variance within each W-sample segment: slow amplitude modulation is intermittency, not nonlinearity."""
    k = len(v) // w
    s = v[:k * w].reshape(k, w)
    s = (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)
    return s.ravel()


def _nonlinearity(x, y):
    """Nonlinear share in the better-predicted direction (larger cross-validated full R^2)."""
    (la, fa), (lb, fb) = _volterra(x, y), _volterra(y, x)
    lin, full = (la, fa) if fa >= fb else (lb, fb)
    return (1 - max(lin, 0) / full if full > 0 else np.nan), max(fa, fb)


def _nonlinearity_floor(x, y):
    return max(_volterra(x, y, True), _volterra(y, x, True))


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

def _clip(v):
    return float(np.clip(v, 0, 1)) if np.isfinite(v) else np.nan


def estimate(x, y, rng=None, floor=True, w=W, shifts=SHIFTS):
    """All axes for one pair. ``floor`` tests each axis's dependence against ``shifts`` circular shifts of y."""
    xr, yr = standardise(x), standardise(y)
    xg, yg = gaussianise(x), gaussianise(y)
    xl, yl = local_standardise(xg, w), local_standardise(yg, w)
    xlr, ylr = local_standardise(xr, w), local_standardise(yr, w)
    xw = whiten(xg)
    lag, coh, stab, stead, local = _spectral(xw, whiten(yg), w)
    nl, pred = _nonlinearity(xl, yl)
    nlr, predr = _nonlinearity(xlr, ylr)
    d, gsum = _direction(xg, yg)
    out = dict(nonlinearity=nl, lag=lag, lag_variability=stab, intermittency=stead, direction=d, nonlinearity_raw=nlr)
    out = {k: _clip(v) for k, v in out.items()}
    out.update(pearson=float(np.corrcoef(xr, yr)[0, 1]), coherence=float(coh / (w // 2 - 1)),
               local_coupling=float(local / ((w // 4 + 1) // 2 - 1)))
    if floor:
        obs = dict(nonlinearity=pred, nonlinearity_raw=predr, lag=coh, lag_variability=local, intermittency=local, direction=gsum)
        null = {k: [] for k in obs}
        rng = rng or np.random.default_rng(0)
        n = len(x)
        for s in rng.integers(w, n - w, shifts):
            ys = np.roll(yg, s)
            _, cs, _, _, loc = _spectral(xw, whiten(ys), w)
            null['lag'].append(cs); null['lag_variability'].append(loc); null['intermittency'].append(loc)
            null['nonlinearity'].append(_nonlinearity_floor(xl, np.roll(yl, s)))
            null['nonlinearity_raw'].append(_nonlinearity_floor(xlr, np.roll(ylr, s)))
            null['direction'].append(_direction_floor(xg, ys))
        for k in obs:
            p = (1 + np.sum(np.array(null[k]) >= obs[k])) / (1 + shifts)
            out[f'{k}_defined'] = bool(p <= ALPHA)
            if not out[f'{k}_defined']:
                out[k] = np.nan
    return out
