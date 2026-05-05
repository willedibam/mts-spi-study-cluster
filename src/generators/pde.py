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
    n_modes: int = 4,
    ic_decay: float = 1.5,
    noise_std: float = 0.0,
    seed: int | None = None,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    1D wave equation: d^2z/dt^2 = c^2 * d^2z/du^2 on a periodic domain.

    Per-instance variability comes from a random multi-mode Fourier IC:
        z(u, 0) = sum_{k=1..K} a_k * cos(2*pi*k*u/L + phi_k),
        a_k ~ N(0, 1) / k**ic_decay,   phi_k ~ U(0, 2*pi),
    with zero initial velocity. Two instances are not circular shifts of
    one another, which is what made a single-pulse IC degenerate.

    Knobs:
      - c          : wave speed (rescales physical time only; does not change
                     the visible MTS shape because dt = 0.2*dx/c).
      - n_modes    : number of Fourier modes in the IC. Capped at M//2 (Nyquist).
      - ic_decay   : amplitude decay across modes (a_k ~ 1 / k**ic_decay).
      - noise_std  : optional additive observation noise per channel/timestep
                     (Gaussian, applied AFTER the wave evolution but BEFORE
                     z-scoring). Default 0.0 (off).
    """
    rng = _resolve_rng(seed, rng)

    dx = 1.0 / M
    dt = 0.2 * dx / c
    coeff = (c * dt / dx) ** 2

    coords = np.arange(M, dtype=float) / M
    K = int(min(max(1, n_modes), max(1, M // 2)))
    ks = np.arange(1, K + 1, dtype=float)
    amps = rng.normal(0.0, 1.0, size=K) / (ks ** ic_decay)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=K)

    z_prev = np.zeros(M, dtype=float)
    for k, a, ph in zip(ks, amps, phases):
        z_prev += a * np.cos(2.0 * np.pi * k * coords + ph)
    s = np.std(z_prev)
    if s > 0:
        z_prev = z_prev / s

    # Zero initial velocity: z_curr = z_prev + 0.5 * dt^2 * c^2 * lap(z_prev)
    z_curr = z_prev + 0.5 * coeff * _laplacian_1d(z_prev)

    samples = np.zeros((T, M), dtype=float)
    samples[0] = z_prev
    if T > 1:
        samples[1] = z_curr

    for t in range(2, T):
        lap = _laplacian_1d(z_curr)
        z_next = 2.0 * z_curr - z_prev + coeff * lap
        samples[t] = z_next
        z_prev, z_curr = z_curr, z_next

    if noise_std > 0:
        samples = samples + rng.normal(0.0, noise_std, size=samples.shape)

    return _maybe_zscore(samples, zscore=zscore)


def generate_wave_1d_pulse(
    M: int,
    T: int,
    *,
    c: float = 10.0,
    sigma_factor: float = 0.15,
    direction: str = "random",
    noise_std: float = 0.0,
    seed: int | None = None,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    Single Gaussian pulse propagating unidirectionally on a periodic 1D domain
    under the wave equation z_tt = c^2 z_uu.

    Pure unidirectional propagation is enforced by the d'Alembert relation:
        for a right-moving solution z(u, t) = f(u - c t),
        \\dot z(u, 0) = -c * d/du z(u, 0);
        for a left-moving solution swap the sign.
    Without this, a stationary Gaussian IC splits into two counter-propagating
    half-amplitude pulses (which is what the old buggy generator did).

    Per-instance variability:
      - random pulse centre (uniform on the ring),
      - random direction (left or right) when direction='random'.
    Note: this generator has fundamentally low per-instance entropy on a
    translation-equivariant periodic system. All right-moving instances are
    circular shifts of one another, so SPI-feature-space they reduce to <=2
    effective classes. Use as a visual / case-study generator, NOT as an
    anchor for 50-instance SPI fingerprinting (use generate_wave_1d for that).

    Knobs:
      - c            : wave speed (rescales physical time only, dt = 0.2*dx/c).
      - sigma_factor : Gaussian width as a fraction of M. Default 0.15 ->
                       sigma = 0.15 * M grid cells. M >= 20 recommended for
                       resolution; below that the pulse is < 3 cells wide.
      - direction    : 'right', 'left', 'random' (unidirectional), or
                       'standing' (zero velocity -> splits into both halves
                       and produces the diamond / X interference lattice
                       characteristic of a localized IC released at rest).
      - noise_std    : optional additive observation noise.
    """
    if direction not in {"right", "left", "random", "standing"}:
        raise ValueError(
            f"direction must be 'right', 'left', 'random', or 'standing'. Got {direction!r}."
        )
    rng = _resolve_rng(seed, rng)

    dx = 1.0 / M
    dt = 0.2 * dx / c
    coeff = (c * dt / dx) ** 2

    coords = np.arange(M, dtype=float)
    centre = float(rng.uniform(0.0, M))
    sigma_cells = max(1.0, sigma_factor * M)

    delta = coords - centre
    delta = (delta + M / 2.0) % M - M / 2.0

    z_prev = np.exp(-(delta ** 2) / (2.0 * sigma_cells ** 2))
    z_prev = z_prev / np.max(np.abs(z_prev))

    if direction == "standing":
        # \dot z(0) = 0 -> pulse splits into two counter-propagating halves,
        # producing a diamond / X interference lattice on the periodic ring.
        v0 = np.zeros_like(z_prev)
    else:
        if direction == "random":
            sign = -1.0 if rng.uniform() < 0.5 else 1.0  # -1 = right, +1 = left
        else:
            sign = -1.0 if direction == "right" else 1.0
        # Centred FD spatial derivative on periodic grid.
        dz_du = (np.roll(z_prev, -1) - np.roll(z_prev, 1)) / (2.0 * dx)
        v0 = sign * c * dz_du

    z_curr = z_prev + dt * v0 + 0.5 * coeff * _laplacian_1d(z_prev)

    samples = np.zeros((T, M), dtype=float)
    samples[0] = z_prev
    if T > 1:
        samples[1] = z_curr

    for t in range(2, T):
        lap = _laplacian_1d(z_curr)
        z_next = 2.0 * z_curr - z_prev + coeff * lap
        samples[t] = z_next
        z_prev, z_curr = z_curr, z_next

    if noise_std > 0:
        samples = samples + rng.normal(0.0, noise_std, size=samples.shape)

    return _maybe_zscore(samples, zscore=zscore)


def generate_wave_2d(
    M: int,
    T: int,
    *,
    c: float = 10.0,
    n_modes: int = 3,
    ic_decay: float = 1.5,
    noise_std: float = 0.0,
    seed: int | None = None,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    2D wave equation on a periodic side x side grid. M must be a perfect square.

    Per-instance variability comes from a random 2D Fourier IC:
        z(u, v, 0) = sum_{(kx, ky) != (0,0)} a_{kx,ky}
                       * cos(2*pi*(kx*u + ky*v)/L + phi_{kx,ky}),
        a ~ N(0, 1) / (kx^2 + ky^2)**(ic_decay/2),  phi ~ U(0, 2*pi),
    with zero initial velocity.

    Knobs match generate_wave_1d. n_modes here is the per-axis mode cap
    (so the IC sums (n_modes+1)^2 - 1 modes).
    """
    rng = _resolve_rng(seed, rng)

    side = int(np.sqrt(M))
    if side * side != M:
        raise ValueError(f"Wave 2D generator requires M to be a perfect square. Got M={M}.")

    dx = 1.0 / side
    dt = 0.2 * dx / c
    coeff = (c * dt / dx) ** 2

    coords = np.arange(side, dtype=float) / side
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    K = int(min(max(1, n_modes), max(1, side // 2)))

    z_prev = np.zeros((side, side), dtype=float)
    for kx in range(K + 1):
        for ky in range(K + 1):
            if kx == 0 and ky == 0:
                continue
            mode_norm = float(kx * kx + ky * ky)
            amp = rng.normal(0.0, 1.0) / (mode_norm ** (ic_decay / 2.0))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            z_prev += amp * np.cos(2.0 * np.pi * (kx * X + ky * Y) + phase)
    s = np.std(z_prev)
    if s > 0:
        z_prev = z_prev / s

    z_curr = z_prev + 0.5 * coeff * _laplacian_2d(z_prev)

    samples = np.zeros((T, M), dtype=float)
    samples[0] = z_prev.reshape(-1)
    if T > 1:
        samples[1] = z_curr.reshape(-1)

    for t in range(2, T):
        lap = _laplacian_2d(z_curr)
        z_next = 2.0 * z_curr - z_prev + coeff * lap
        samples[t] = z_next.reshape(-1)
        z_prev, z_curr = z_curr, z_next

    if noise_std > 0:
        samples = samples + rng.normal(0.0, noise_std, size=samples.shape)

    return _maybe_zscore(samples, zscore=zscore)
