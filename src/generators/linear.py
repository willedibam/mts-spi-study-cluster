from __future__ import annotations

import numpy as np

from ._common import _maybe_zscore, _resolve_rng


def generate_varma(
    M: int,
    T: int,
    phi: float = 0.6,
    coupling: float = 0.4,
    ma_phi: float = 0.2,
    ma_coupling: float = 0.1,
    noise_std: float = 0.1,
    transients: int = 100,
    target_rho: float = 0.99,
    topology: str = "ring-symmetric",
    rng=None,
    zscore: bool = True,
):
    """
    Generates VARMA(p,q) process with specified topology.

    Supported topologies:
      - 'ring-symmetric': Standard diffuse coupling (i connected to i-1 AND i+1).
      - 'ring-unidirectional': Advective coupling (i connected to i+1 only).
      - 'all-to-all': Each channel receives the same coupling from every other channel.
    """
    rng = _resolve_rng(None, rng)
    I = np.eye(M)
    right = np.roll(I, 1, axis=1)
    left = np.roll(I, -1, axis=1)
    if topology == "ring-symmetric":
        neighbors = left + right
    elif topology == "ring-unidirectional":
        neighbors = left
    elif topology == "all-to-all":
        neighbors = np.ones((M, M)) - I
    else:
        raise ValueError(f"Unknown topology: {topology}")
    A = phi * I + coupling * neighbors
    ev = np.linalg.eigvals(A)
    sr = np.max(np.abs(ev))
    if sr >= target_rho:
        A = A * (target_rho / sr)
    B = ma_phi * I + ma_coupling * neighbors
    steps = transients + T
    X = np.zeros((steps, M), float)
    eps = rng.normal(0.0, noise_std, size=(steps, M))
    for t in range(1, steps):
        X[t] = A @ X[t - 1] + eps[t] + B @ eps[t - 1]
    return _maybe_zscore(X[transients:], zscore=zscore)


def generate_varma_shuffled(
    M: int,
    T: int,
    phi: float = 0.6,
    coupling: float = 0.4,
    ma_phi: float = 0.2,
    ma_coupling: float = 0.1,
    noise_std: float = 0.1,
    transients: int = 100,
    target_rho: float = 0.99,
    topology: str = "ring-symmetric",
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    rng = _resolve_rng(None, rng)
    X = generate_varma(
        M=M, T=T,
        phi=phi, coupling=coupling,
        ma_phi=ma_phi, ma_coupling=ma_coupling,
        noise_std=noise_std, transients=transients,
        target_rho=target_rho, topology=topology, rng=rng, zscore=False
    )
    for m in range(M):
        rng.shuffle(X[:, m])
    return _maybe_zscore(X, zscore=zscore)


def generate_gaussian_noise(
    M: int,
    T: int,
    *,
    zscore: bool = True,
    rng=None,
) -> np.ndarray:
    rng = _resolve_rng(None, rng)
    raw_data = rng.normal(size=(T, M))
    return _maybe_zscore(raw_data, zscore=zscore)


def generate_cauchy_noise(
    M: int,
    T: int,
    *,
    zscore: bool = False,
    rng=None,
) -> np.ndarray:
    rng = _resolve_rng(None, rng)
    data = rng.standard_cauchy(size=(T, M))
    return _maybe_zscore(data, zscore=zscore) if zscore else data


def generate_exponential_noise(
    M: int,
    T: int,
    *,
    zscore: bool = False,
    rng=None,
) -> np.ndarray:
    """Standard Exponential distribution (rate parameter gamma = 1)."""
    rng = _resolve_rng(None, rng)
    data = rng.exponential(scale=1.0, size=(T, M))
    return _maybe_zscore(data, zscore=zscore) if zscore else data


def generate_gbm(
    M: int,
    T: int,
    *,
    mu: float | np.ndarray = 0.0,
    sigma: float | np.ndarray = 0.2,
    dt: float = 1.0,
    s0: float | np.ndarray = 1.0,
    rng=None,
    zscore: bool = False,
) -> np.ndarray:
    """
    Simulate M independent geometric Brownian motion paths.

    dS_t / S_t = mu * dt + sigma * dW_t
    Discretised with Euler-Maruyama in log space:
        S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*N(0,1))
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    rng = _resolve_rng(None, rng)

    mu_arr = np.broadcast_to(np.asarray(mu, dtype=float), (M,))
    sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=float), (M,))
    s0_arr = np.broadcast_to(np.asarray(s0, dtype=float), (M,))

    paths = np.zeros((T, M), dtype=float)
    paths[0] = s0_arr

    noise = rng.normal(size=(max(T - 1, 0), M))
    drift = (mu_arr - 0.5 * sigma_arr**2) * dt
    diffusion_scale = sigma_arr * np.sqrt(dt)

    for t in range(1, T):
        increment = drift + diffusion_scale * noise[t - 1]
        paths[t] = paths[t - 1] * np.exp(increment)

    return _maybe_zscore(paths, zscore=zscore)
