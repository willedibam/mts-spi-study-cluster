"""Two-channel test systems: one knob theta per axis, with every other property held fixed.

The knob is mapped so that the analytic value of the system's own axis equals theta (direction excepted:
its value comes from the long-run estimate). x is a unit-variance Gaussian AR process and eta an
independent copy; sigma scales the private noise in y.

- linearity:       y = sqrt(1 - theta) x + sqrt(theta) He2(x) + sigma eta. He2(x) = (x^2 - 1)/sqrt 2 is
                   uncorrelated with x at every lag, so raw nonlinearity = theta at any sigma.
- lag:             y = cos(phi) x + sin(phi) H[x] + sigma eta, sin^2 phi = theta (H: Hilbert transform).
                   Coherence magnitude is fixed; lag share = theta.
- lag_variability: as lag, with phi = pi/4 + s delta_k constant within each W-sample segment, delta_k ~ N(0, 1),
                   1 - exp(-s^2) = theta. Local coupling is unchanged; lag variability = theta.
- intermittency:   y = a_k x + sigma eta, coupling on in a fraction q = 1 - theta of segments, with a and sigma
                   set so Pearson r and var(y) are fixed. Intermittency = theta.
- direction:       bivariate VAR, own AR(1) 0.5, lag-2 cross coupling c split (1 +- theta)/2 between directions.
- independent:     x and eta, independent.

Nuisances, applied in this order: instantaneous mixing (x, y) -> (x + mix y, y + mix x); white measurement
noise of SD xnoise on x only; static distortion y -> sinh(beta y)/beta.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from research.representation.character_estimators.estimators import W

SPECS = dict(ar1=('ar1', .8), broad=('ar1', .5), slow=('ar1', .95), narrow=('ar2', (.95, .1)))
SYSTEMS = ('linearity', 'lag', 'lag_variability', 'intermittency', 'direction')
OWN = dict(linearity='nonlinearity_raw', lag='lag', lag_variability='lag_variability', intermittency='intermittency',
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


def make(system, theta, T, rng, strength=None, spec='ar1', beta=0., mix=0., xnoise=0., w=W):
    """(x, y) for one recording. ``strength`` is sigma, except Pearson r for intermittency and c for direction."""
    if system == 'direction':
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
