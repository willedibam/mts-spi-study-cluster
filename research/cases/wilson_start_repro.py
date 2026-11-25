"""Data-free check of spectral_connectivity's Wilson starting factor.

spectral_connectivity (2.0.1, and its master on 2026-11-26) starts Wilson's
iteration from ``cholesky(G0).swapaxes(-1, -2)``, whose product with its
conjugate transpose is not the zero-lag covariance G0. This feeds the exact
spectrum of a bivariate VAR(1) to the backend under that start and under an
upper-triangular start with psi0 psi0^H = G0, and compares the band-mean
spectral GC of each direction with the analytic value (nitime, true model).
x0 drives x1, so the truth for x1 -> x0 is 0.

Run: python -m research.cases.wilson_start_repro
"""
import logging

import numpy as np
import nitime.algorithms as nt
from scipy.fft import ifft
from spectral_connectivity import minimum_phase_decomposition as mpd
from spectral_connectivity.connectivity import _estimate_spectral_granger_prediction

N_FREQ = 1000
BACKEND_START = mpd._get_initial_conditions


def upper_start(csm):
    """Upper-triangular psi0 with psi0 psi0^H equal to the zero-lag covariance."""
    g0 = ifft(csm, axis=-3)[..., 0:1, :, :].real
    flip = np.eye(g0.shape[-1])[::-1]
    return flip @ np.linalg.cholesky(flip @ g0 @ flip) @ flip


def var1_spectrum(A, C, n=N_FREQ):
    w = 2 * np.pi * np.fft.fftfreq(n)
    H = np.linalg.inv(np.eye(2)[None] - A[None] * np.exp(-1j * w)[:, None, None])
    return (H @ C @ np.conj(np.swapaxes(H, -1, -2)))[None]


def band_gc(csm, start):
    """Mean GC over positive frequencies, (x0 -> x1, x1 -> x0), and factor residual."""
    mpd._get_initial_conditions = start
    try:
        G = mpd.minimum_phase_decomposition(csm)
        power = np.real(np.diagonal(csm, axis1=-2, axis2=-1))
        F = np.asarray(_estimate_spectral_granger_prediction(power, csm, [(0, 1)]))
    finally:
        mpd._get_initial_conditions = BACKEND_START
    resid = np.abs(G @ np.conj(np.swapaxes(G, -1, -2)) - csm).max() / np.abs(csm).max()
    # F[0, f, target, source]
    return np.nanmean(F[0, 1:, 1, 0]), np.nanmean(F[0, 1:, 0, 1]), resid


def main():
    logging.disable(logging.WARNING)
    A = np.array([[0.99, 0.0], [0.05, 0.98]])  # near-unit-root, x0 -> x1
    for rho in (0.0, 0.5, 0.9, 0.97):
        C = np.array([[1.0, rho], [rho, 1.0]])
        _, f01, _, _, _ = nt.granger_causality_xy(A[None], C, n_freqs=N_FREQ)
        csm = var1_spectrum(A, C)
        b = band_gc(csm, BACKEND_START)
        u = band_gc(csm, upper_start)
        print(f"rho={rho}: truth {np.mean(f01):.4f}/0 | backend {b[0]:.4f}/{b[1]:.4f} "
              f"resid {b[2]:.1e} | upper start {u[0]:.4f}/{u[1]:.4f} resid {u[2]:.1e}")


if __name__ == "__main__":
    main()
