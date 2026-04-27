from __future__ import annotations

import numpy as np

from ._common import _maybe_zscore, _resolve_rng


def _laplacian_1d(z: np.ndarray) -> np.ndarray:
    """Finite difference Laplacian in 1D with periodic boundary conditions."""
    return np.roll(z, -1) - 2.0 * z + np.roll(z, 1)


def _laplacian_2d(z: np.ndarray) -> np.ndarray:
    """Finite difference Laplacian in 2D with periodic boundary conditions."""
    grad_u = np.roll(z, -1, axis=0) - 2.0 * z + np.roll(z, 1, axis=0)
    grad_v = np.roll(z, -1, axis=1) - 2.0 * z + np.roll(z, 1, axis=1)
    return grad_u + grad_v


def generate_wave_1d(
    M: int,
    T: int,
    *,
    c: float = 10.0,
    seed: int | None = None,
    rng=None,
    zscore: bool = True,
    center_jitter: float = 0.5,
    sigma_factor: float = 1.0 / 20.0,
    vel_std: float = 0.0,
    vel_modes: int = 6,
    vel_decay: float = 2.0,
) -> np.ndarray:
    """
    Simulates the 1D Wave Equation: d^2z/dt^2 = c^2 * d^2z/du^2

    Periodic BC. Default IC: Gaussian pulse, sigma = M * sigma_factor (BDT default 1/20).

    Per-instance variability sources:
      - center_jitter: pulse centre ~ U(M/2 - jit*M, M/2 + jit*M), wrapped periodically.
      - vel_std: optional bandlimited random IC velocity (off by default).
    """
    rng = _resolve_rng(seed, rng)

    dx = 1.0 / M
    dt = 0.2 * dx / c
    coeff = (c * dt / dx) ** 2

    coords = np.arange(M, dtype=float)
    center = M / 2.0
    if center_jitter > 0:
        center = (center + rng.uniform(-center_jitter * M, center_jitter * M)) % M
    sigma = M * sigma_factor

    # Periodic distance to centre so off-centre pulses don't create a boundary discontinuity.
    delta = coords - center
    delta = (delta + M / 2.0) % M - M / 2.0

    z_prev = np.exp(-(delta ** 2) / (2.0 * sigma ** 2))
    z_prev = z_prev / np.max(np.abs(z_prev))

    v0 = np.zeros(M, dtype=float)
    if vel_std > 0:
        x = coords / M
        K = int(min(max(1, vel_modes), M // 2))
        ks = np.arange(1, K + 1, dtype=float)

        phases = rng.uniform(0.0, 2.0 * np.pi, size=K)
        amps = rng.normal(0.0, 1.0, size=K) / (ks ** vel_decay)

        for k, a, ph in zip(ks, amps, phases):
            v0 += a * np.sin(2.0 * np.pi * k * x + ph)

        v0 -= v0.mean()
        s = np.std(v0)
        if s > 0:
            v0 = (v0 / s) * vel_std

    lap_prev = _laplacian_1d(z_prev)
    z_curr = z_prev + dt * v0 + 0.5 * coeff * lap_prev

    samples = np.zeros((T, M), dtype=float)
    samples[0] = z_prev
    if T > 1:
        samples[1] = z_curr

    for t in range(2, T):
        lap = _laplacian_1d(z_curr)
        z_next = 2.0 * z_curr - z_prev + coeff * lap
        samples[t] = z_next
        z_prev, z_curr = z_curr, z_next

    return _maybe_zscore(samples, zscore=zscore)


def generate_wave_2d(
    M: int,
    T: int,
    *,
    c: float = 10.0,
    seed: int | None = None,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    Simulates the 2D Wave Equation: d^2z/dt^2 = c^2 * (d^2z/du^2 + d^2z/dv^2)

    Periodic boundary conditions. M must be a perfect square.
    """
    rng = _resolve_rng(seed, rng)

    side = int(np.sqrt(M))
    if side * side != M:
        raise ValueError(f"Wave 2D generator requires M to be a perfect square. Got M={M}.")

    dx = 1.0 / side
    dt = 0.2 * dx / c
    coeff = (c * dt / dx) ** 2

    x = np.arange(side, dtype=float)
    X, Y = np.meshgrid(x, x, indexing="ij")
    center = side / 2.0
    sigma = M / 20.0

    dist_sq = (X - center) ** 2 + (Y - center) ** 2
    z_prev = np.exp(-(dist_sq) / (2.0 * sigma**2))
    z_prev = z_prev / np.max(np.abs(z_prev))

    lap_prev = _laplacian_2d(z_prev)
    z_curr = z_prev + 0.5 * coeff * lap_prev

    samples = np.zeros((T, M), dtype=float)
    samples[0] = z_prev.reshape(-1)
    if T > 1:
        samples[1] = z_curr.reshape(-1)

    for t in range(2, T):
        lap = _laplacian_2d(z_curr)
        z_next = 2.0 * z_curr - z_prev + coeff * lap
        samples[t] = z_next.reshape(-1)
        z_prev, z_curr = z_curr, z_next

    return _maybe_zscore(samples, zscore=zscore)
