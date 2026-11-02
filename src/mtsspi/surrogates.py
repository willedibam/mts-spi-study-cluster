"""Multivariate linear surrogates for character tests.

Both generators add one random phase per frequency to every channel, so all
auto- and cross-spectra, and hence every linear lag and coherence relation, are
preserved (Prichard and Theiler 1994). What they destroy is structure beyond a
stationary linear Gaussian process: non-Gaussian dependence, phase-amplitude
interactions and nonstationary or intermittent coupling.

``multivariate_iaaft`` also restores each channel's exact value distribution
(Schreiber and Schmitz 2000), so heavy-tailed marginals do not masquerade as
nonlinear coupling; its spectra are then matched approximately, not exactly.
Both assume periodic boundaries: windows whose ends differ strongly gain a
spurious discontinuity.
"""
import numpy as np


def phase_randomized(x, rng):
    """Exact cross-spectrum surrogate of a (T, M) array."""
    x = np.asarray(x, dtype=np.float64)
    f = np.fft.rfft(x, axis=0)
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, f.shape[0]))
    phase[0] = 1
    if len(x) % 2 == 0:
        phase[-1] = 1
    return np.fft.irfft(f * phase[:, None], n=len(x), axis=0)


def multivariate_iaaft(x, rng, iterations=200, tolerance=1e-10, final='marginal'):
    """Amplitude-adjusted multivariate surrogate of a (T, M) array.

    Returns the surrogate and a diagnostic dict. ``final='marginal'`` ends on
    the rank step (exact marginals, approximate spectra); ``final='spectrum'``
    ends on the Fourier step (exact auto- and cross-spectra, approximate
    marginals). Strongly non-Gaussian signals leave a mismatch either way, so
    the two variants bracket the linear null. Relative phases between channels
    follow the original.
    """
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    ordered = np.sort(x, axis=0)
    target = np.fft.rfft(x, axis=0)
    amplitude, phase0 = np.abs(target), np.angle(target)
    s = np.take_along_axis(x, np.argsort(rng.random(x.shape), axis=0), axis=0)
    previous = None
    for iteration in range(iterations):
        current = np.angle(np.fft.rfft(s, axis=0))
        # Common rotation alpha(f) closest to the current phases, keeping the
        # original phase differences between channels.
        delta = np.exp(1j * (current - phase0)).sum(1)
        alpha = np.angle(delta)
        s = np.fft.irfft(amplitude * np.exp(1j * (phase0 + alpha[:, None])), n=T, axis=0)
        ranks = np.argsort(np.argsort(s, axis=0), axis=0)
        s = np.take_along_axis(ordered, ranks, axis=0)
        if previous is not None and np.array_equal(ranks, previous):
            break
        previous = ranks
    if final == 'spectrum':
        current = np.angle(np.fft.rfft(s, axis=0))
        alpha = np.angle(np.exp(1j * (current - phase0)).sum(1))
        s = np.fft.irfft(amplitude * np.exp(1j * (phase0 + alpha[:, None])), n=T, axis=0)
    elif final != 'marginal':
        raise ValueError(final)
    achieved = np.abs(np.fft.rfft(s, axis=0))
    spectrum_error = float(np.linalg.norm(achieved - amplitude) / np.linalg.norm(amplitude))
    cov_x, cov_s = np.corrcoef(x.T), np.corrcoef(s.T)
    off = ~np.eye(x.shape[1], dtype=bool)
    marginal_error = float(np.max(np.abs(np.sort(s, axis=0) - ordered)) / np.max(np.abs(ordered)))
    return s, dict(iterations=iteration + 1, final=final, amplitude_relative_error=spectrum_error, marginal_max_relative_error=marginal_error,
                   correlation_max_abs_error=float(np.max(np.abs(cov_x - cov_s)[off])) if x.shape[1] > 1 else 0.0)
