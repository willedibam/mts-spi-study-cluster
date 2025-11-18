from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
from numpy.random import default_rng
from scipy.stats import zscore

GeneratorFn = Callable[..., np.ndarray]


def _global_rng():
    return default_rng(123456789)


def _resolve_rng(seed: int | None, rng=None):
    if rng is not None:
        return rng
    if seed is None:
        return _global_rng()
    return default_rng(seed)


def generate_var(
    M: int,
    T: int,
    phi: float = 0.95,
    coupling: float = 0.8,
    noise_std: float = 0.1,
    transients: int = 100,
    rng=None,
) -> np.ndarray:
    rng = _resolve_rng(None, rng)
    A = np.eye(M) * phi + (coupling / M) * (np.ones((M, M)) - np.eye(M))
    ev = np.linalg.eigvals(A)
    sr = np.max(np.abs(ev))
    if sr >= 0.98:
        A = A / (1.05 * sr)
    steps = transients + T
    X = np.zeros((steps, M), float)
    eps = rng.normal(0.0, noise_std, size=(steps, M))
    for t in range(1, steps):
        X[t] = A @ X[t - 1] + eps[t]
    return zscore(X[transients:], axis=0)


def generate_gaussian_noise(M: int, T: int, rng=None) -> np.ndarray:
    rng = _resolve_rng(None, rng)
    return rng.normal(size=(T, M))


def generate_cauchy_noise(M: int, T: int, rng=None) -> np.ndarray:
    rng = _resolve_rng(None, rng)
    return rng.standard_cauchy(size=(T, M))


def generate_cml_logistic(
    M: int,
    T: int,
    alpha: float = 1.7522,
    eps: float = 0.00115,
    delta: int = 12,
    transients: int = 100,
    rng=None,
) -> np.ndarray:
    rng = _resolve_rng(None, rng)

    def logistic(x, a):
        return 1 - a * x**2

    def iterate_map(x, epsilon, f):
        fx = f(x)
        left = np.roll(fx, 1)
        right = np.roll(fx, -1)
        return (1 - epsilon) * fx + epsilon / 2 * (left + right)

    f = lambda x: logistic(x, alpha)
    steps = (transients + T) * delta
    X = rng.uniform(-1, 1, size=M)
    Y = np.zeros((steps, M))
    for t in range(steps):
        Y[t] = X
        X = iterate_map(X, eps, f)
    output = Y[(transients * delta) :: delta, :]
    return zscore(output[:T], axis=0)

def _build_kuramoto_adjacency(
    M: int,
    scheme: str = "bidirectional_list",
    k_ring: int = 1,
) -> np.ndarray:
    """
    Build a boolean adjacency matrix a_ij for the Kuramoto model.

    scheme:
      - "all_to_all": every oscillator connected to every other.
      - "bidirectional_list": 1D ring, each node to k_ring neighbours on each side.
      - "grid_four": 2D torus, each node to its 4 nearest neighbours;
                     requires M to be a perfect square.
    """
    scheme = scheme.lower()
    A = np.zeros((M, M), float)

    if scheme in {"all_to_all", "all-to-all"}:
        A[:] = 1.0
        np.fill_diagonal(A, 0.0)

    elif scheme in {"bidirectional_list", "ring", "list"}:
        for i in range(M):
            for d in range(1, k_ring + 1):
                A[i, (i + d) % M] = 1.0
                A[i, (i - d) % M] = 1.0

    elif scheme in {"grid_four", "grid-4", "grid"}:
        side = int(np.sqrt(M))
        if side * side != M:
            raise ValueError(
                f"grid_four scheme requires M to be a perfect square, got M={M}"
            )
        for idx in range(M):
            r, c = divmod(idx, side)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = (r + dr) % side
                nc = (c + dc) % side
                j = nr * side + nc
                A[idx, j] = 1.0
    else:
        raise ValueError(f"Unknown Kuramoto coupling scheme '{scheme}'")

    return A

def generate_kuramoto(
    M: int,
    T: int,
    dt: float = 0.002,
    K: float = 1.5,
    k_ring: int = 1,
    omega_mean: float = 2 * np.pi * 0.1,
    omega_std: float = 0.01,
    eta: float = 0.0,
    transients: int = 2000,
    output: str = "sin",
    coupling_scheme: str = "bidirectional_list",
    rng=None,
) -> np.ndarray:
    """
    Kuramoto network:

        dθ_i/dt = ω_i + K * (1/deg_i) * sum_j a_ij sin(θ_j - θ_i) + noise

    where a_ij is boolean adjacency determined by `coupling_scheme`
    and deg_i = sum_j a_ij.

    We then output z_i(t) = sin θ_i(t) (or cos/θ directly) and z-score per channel.
    """
    rng = _resolve_rng(None, rng)

    # Boolean adjacency a_ij
    A = _build_kuramoto_adjacency(M, scheme=coupling_scheme, k_ring=k_ring)
    degree = A.sum(axis=1)
    # Avoid division by zero if some node has no neighbours
    inv_degree = np.where(degree > 0, 1.0 / degree, 0.0)

    # Initial phases and natural frequencies
    theta = rng.uniform(0.0, 2.0 * np.pi, size=M)
    omega = rng.normal(omega_mean, omega_std, size=M)

    steps = transients + T
    Y = np.zeros((steps, M), float)

    for t in range(steps):
        # Record observable z_i(t)
        if output == "sin":
            Y[t] = np.sin(theta)
        elif output == "cos":
            Y[t] = np.cos(theta)
        else:
            Y[t] = theta

        # Kuramoto coupling term
        phase_diff = theta[None, :] - theta[:, None]    # θ_j - θ_i in [i,j] layout
        coupling_term = (A * np.sin(phase_diff)).sum(axis=1)

        dtheta = omega + K * inv_degree * coupling_term
        noise = eta * np.sqrt(dt) * rng.normal(size=M)
        theta = np.mod(theta + dtheta * dt + noise, 2.0 * np.pi)

    return zscore(Y[transients:], axis=0)


GENERATOR_REGISTRY: Dict[str, GeneratorFn] = {
    "var": generate_var,
    "cml_logistic": generate_cml_logistic,
    "kuramoto": generate_kuramoto,
    "gaussian_noise": generate_gaussian_noise,
    "cauchy_noise": generate_cauchy_noise,
}


def available_generators() -> list[str]:
    return sorted(GENERATOR_REGISTRY.keys())


def generate_series(name: str, *, seed: int | None = None, **params: Any) -> np.ndarray:
    if name not in GENERATOR_REGISTRY:
        raise KeyError(f"Unknown generator '{name}'. Known: {available_generators()}")
    rng = _resolve_rng(seed)
    gen = GENERATOR_REGISTRY[name]
    return gen(rng=rng, **params)

