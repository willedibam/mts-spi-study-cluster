"""Two-channel test systems: one knob theta per axis, with every other property held fixed.

The knob is mapped so that the analytic value of the system's own axis equals theta (direction excepted:
its value comes from the long-run estimate). x is a unit-variance Gaussian AR process and eta an
independent copy; sigma scales the private noise in y.

- linearity:       y = sqrt(1 - theta) x + sqrt(theta) He2(x) + sigma eta. He2(x) = (x^2 - 1)/sqrt 2 is
                   uncorrelated with x at every lag, so raw nonlinearity = theta at any sigma.
- lag:             y = cos(phi) x + sin(phi) H[x] + sigma eta, sin^2 phi = theta (H: Hilbert transform).
                   Coherence magnitude is fixed; lag share = theta.
- lag_variability: as lag, with phi = pi/4 + s delta_k constant within each W-sample segment, delta_k ~ N(0, 1),
                   1 - exp(-s^2) = theta. Local coupling is unchanged; variability = theta.
- intermittency:   y = a_k x + sigma eta, coupling on in a fraction q = 1 - theta of segments, with a and sigma
                   set so Pearson r and var(y) are fixed. Variability = theta.
- direction:       bivariate VAR, own AR(1) 0.5, lag-2 cross coupling c split (1 +- theta)/2 between directions.
- independent:     x and eta, independent.

Nuisances, applied in this order: a fixed delay of y behind x by ``delay`` samples; common amplitude
modulation of both channels by exp(ampmod z_k), z_k ~ N(0, 1) per W-sample segment; instantaneous mixing
(x, y) -> (x + mix y, y + mix x); white measurement noise of SD xnoise on x only; static distortion
y -> sinh(beta y)/beta.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from research.representation.character_estimators.estimators import W

SPECS = dict(ar1=('ar1', .8), broad=('ar1', .5), slow=('ar1', .95), narrow=('ar2', (.95, .1)),
             vslow=('ar2', (.99, .02)))
SYSTEMS = ('linearity', 'lag', 'lag_variability', 'intermittency', 'direction')
OWN = dict(linearity='nonlinearity_raw', lag='lag', lag_variability='variability', intermittency='variability',
           direction='direction')
BURN = 1000


def ar(T, spec, rng):
    kind, par = SPECS[spec]
    if kind == 'ar1':
        a = [1, -par]
    else:
        r, f = par
        a = [1, -2 * r * np.cos(2 * np.pi * f), r ** 2]
    h = signal.lfilter([1], a, np.r_[1, np.zeros(4999)])
    x = signal.lfilter([1], a, rng.standard_normal(T + BURN))[BURN:]
    return x / np.sqrt((h ** 2).sum())


def segments(T, w=W):
    return np.minimum(np.arange(T) // w, T // w - 1)


def make(system, theta, T, rng, strength=None, spec='ar1', beta=0., mix=0., xnoise=0., delay=0, ampmod=0., w=W):
    """(x, y) for one recording. ``strength`` is sigma, except Pearson r for intermittency and c for direction."""
    if delay:
        x, y = make(system, theta, T + delay, rng, strength, spec, w=w)
        x, y = x[delay:], y[:T]  # y(t) now depends on x(t - delay)
    elif system == 'direction':
        x, y = _var(theta, T, rng, .6 if strength is None else strength)
    else:
        x, eta = ar(T, spec, rng), ar(T, spec, rng)
        sigma = 1. if strength is None else strength
        if system == 'independent':
            y = eta
        elif system == 'linearity':
            y = np.sqrt(1 - theta) * x + np.sqrt(theta) * (x ** 2 - 1) / np.sqrt(2) + sigma * eta
        elif system in ('lag', 'lag_variability'):
            h = signal.hilbert(x).imag
            if system == 'lag':
                phi = np.arcsin(np.sqrt(theta))
            else:
                s = np.sqrt(-np.log1p(-theta))
                phi = (np.pi / 4 + s * rng.standard_normal(T // w))[segments(T, w)]
            y = np.cos(phi) * x + np.sin(phi) * h + sigma * eta
        elif system == 'intermittency':
            r = .45 if strength is None else strength
            k = T // w
            on = np.zeros(k, bool); on[rng.choice(k, max(1, int(round((1 - theta) * k))), replace=False)] = True
            q = on.mean()
            if r ** 2 / q > 1:
                raise ValueError('intermittency: Pearson r too large for this theta')
            a = np.where(on, r / q, 0.)[segments(T, w)]
            y = a * x + np.sqrt(1 - r ** 2 / q) * eta
        else:
            raise ValueError(system)
    if ampmod:
        e = np.exp(ampmod * rng.standard_normal(T // w + 1))[np.arange(T) // w]
        x, y = e * x, e * y
    if mix:
        x, y = x + mix * y, y + mix * x
    if xnoise:
        x = x + xnoise * rng.standard_normal(T)
    if beta:
        y = np.sinh(beta * y) / beta
    return x, y


def _var(theta, T, rng, c, a=.5):
    cxy, cyx = c * (1 + theta) / 2, c * (1 - theta) / 2
    n = T + BURN
    e = rng.standard_normal((n, 2))
    z = np.zeros((n, 2))
    for t in range(2, n):
        z[t, 0] = a * z[t - 1, 0] + cyx * z[t - 2, 1] + e[t, 0]
        z[t, 1] = a * z[t - 1, 1] + cxy * z[t - 2, 0] + e[t, 1]
    z = z[BURN:]
    return z[:, 0], z[:, 1]


def analytic(system, theta):
    """Closed-form value of the system's own axis at base settings, where one exists."""
    return None if system == 'direction' else theta


# ---- transfer family ------------------------------------------------------------------------------------
# Built so the generators do not share the estimators' construction: oscillatory (narrowband) signals with
# broadband private noise, a nonlinearity outside the Volterra basis, conventional (time-domain) delays, a
# smooth time-varying delay, and switching times independent of the W-sample grid. Truths come from
# integrals over the known spectrum (``transfer_truth``), not from the estimators.

TRANSFER = ('rectifier', 'delay', 'warp', 'telegraph', 'direction_nl')
OWN_TRANSFER = dict(rectifier='nonlinearity_raw', delay='lag', warp='variability', telegraph='variability',
                    direction_nl='direction')
F0 = .1          # carrier of the 'narrow' spec, cycles per sample
WARP_TAU = 2 * W  # correlation time of the time-varying delay, samples
ON_DWELL = 2 * W  # mean dwell of the coupled state, samples


def _shift(v, d):
    """Delay by d samples (fractional allowed): exact for a band-limited signal."""
    f = np.fft.rfftfreq(len(v))
    return np.fft.irfft(np.fft.rfft(v) * np.exp(-2j * np.pi * f * d), n=len(v))


def delay_for(theta):
    """Delay whose phase at the carrier gives a quadrature share theta: sin^2(2 pi F0 d) = theta."""
    return np.arcsin(np.sqrt(theta)) / (2 * np.pi * F0)


def jitter_for(theta):
    """SD of the time-varying delay with 1 - exp(-(2 pi F0 s)^2) = theta at the carrier."""
    return np.sqrt(-np.log1p(-theta)) / (2 * np.pi * F0)


def make_transfer(system, theta, T, rng, strength=None):
    """(x, y): x a narrowband oscillation (spec 'narrow'), eta white; ``strength`` is sigma, except Pearson r
    for telegraph and c for direction_nl."""
    from scipy.interpolate import CubicSpline
    if system == 'direction_nl':
        return _var_nl(theta, T, rng, .6 if strength is None else strength)
    x, eta = ar(T, 'narrow', rng), rng.standard_normal(T)
    sigma = 1. if strength is None else strength
    if system == 'independent':
        return x, eta
    if system == 'rectifier':
        # |x| is even, so uncorrelated with x at every lag; standardised, raw nonlinearity = theta exactly. For
        # an oscillation it carries the envelope and the second harmonic: cross-frequency coupling.
        h = (np.abs(x) - np.sqrt(2 / np.pi)) / np.sqrt(1 - 2 / np.pi)
        return x, np.sqrt(1 - theta) * x + np.sqrt(theta) * h + sigma * eta
    if system == 'delay':
        return x, _shift(x, delay_for(theta)) + sigma * eta
    if system == 'warp':
        z = signal.lfilter([np.sqrt(1 - np.exp(-2 / WARP_TAU))], [1, -np.exp(-1 / WARP_TAU)], rng.standard_normal(T + BURN))[BURN:]
        t = np.arange(T)
        return x, CubicSpline(t, x)(np.clip(t - jitter_for(theta) * z, 0, T - 1)) + sigma * eta
    if system == 'telegraph':
        r, q = (.4 if strength is None else strength), 1 - theta
        on = np.ones(T, bool)
        if theta > 0:  # two-state Markov chain, off-fraction theta, switching independent of the W grid
            p_off, state = 1 / ON_DWELL, rng.random() < q
            p_on = p_off * q / theta
            u = rng.random(T)
            for i in range(T):
                on[i] = state
                state = (u[i] >= p_off) if state else (u[i] < p_on)
        a = np.where(on, r / q, 0.)
        return x, a * x + np.sqrt(1 - r ** 2 / q) * eta
    raise ValueError(system)


def _var_nl(theta, T, rng, c, a=.5):
    """As _var, with the cross terms through tanh: nonlinear directed coupling."""
    cxy, cyx = c * (1 + theta) / 2, c * (1 - theta) / 2
    n = T + BURN
    e = rng.standard_normal((n, 2))
    z = np.zeros((n, 2))
    for t in range(2, n):
        z[t, 0] = a * z[t - 1, 0] + cyx * np.tanh(2 * z[t - 2, 1]) + e[t, 0]
        z[t, 1] = a * z[t - 1, 1] + cxy * np.tanh(2 * z[t - 2, 0]) + e[t, 1]
    z = z[BURN:]
    return z[:, 0], z[:, 1]


def _coherence_narrow(sigma, n=20000):
    """Frequencies in (0, 1/2) and the squared coherence S_x / (S_x + sigma^2) of x with x + sigma eta."""
    kind, (r, f0) = SPECS['narrow']
    a = [1, -2 * r * np.cos(2 * np.pi * f0), r ** 2]
    h = signal.lfilter([1], a, np.r_[1, np.zeros(4999)])
    f = (np.arange(n) + .5) / (2 * n)
    _, H = signal.freqz([1], a, worN=2 * np.pi * f)
    S = np.abs(H) ** 2 / (h ** 2).sum()
    return f, S / (S + sigma ** 2)


def _quarter_pairs(kernel):
    """Mean of kernel(|t - t'|) over sample pairs in distinct quarters of a W-sample segment: the resolution at
    which variability measures local coupling power (estimators._local)."""
    from research.representation.character_estimators.estimators import QUARTERS
    t, q = np.arange(W // 4), W // 4
    return float(np.mean([kernel(np.abs((b * q + t)[None] - (a * q + t)[:, None])).mean() for a, b in QUARTERS]))


def transfer_truth(system, theta, sigma=1.):
    """Population value of the system's own axis, from the generator and the axis definition (integrals over
    the coherence spectrum, and for variability the coupling's autocorrelation at the quarter resolution). None
    where there is no closed form (direction_nl)."""
    f, coh = _coherence_narrow(sigma)
    if system == 'rectifier':
        return theta
    if system == 'delay':
        return float((coh * np.sin(2 * np.pi * f * delay_for(theta)) ** 2).sum() / coh.sum())
    if system == 'telegraph':  # E a(t) a(t + d) = g^2 (q^2 + q (1 - q) exp(-d / tau)), tau = ON_DWELL theta
        if theta == 0:
            return 0.
        q = 1 - theta
        return 1 - q ** 2 / (q ** 2 + q * (1 - q) * _quarter_pairs(lambda d: np.exp(-d / (ON_DWELL * theta))))
    if system == 'warp':  # E exp(i 2 pi f (tau(t) - tau(t + d))) = exp(-v (1 - rho(d))), v = (2 pi f s)^2
        v = (2 * np.pi * f[::50] * jitter_for(theta)) ** 2
        c = coh[::50]
        local = np.array([_quarter_pairs(lambda d: np.exp(-u * (1 - np.exp(-d / WARP_TAU)))) for u in v])
        return float(1 - (c * np.exp(-v)).sum() / (c * local).sum())
    return None
