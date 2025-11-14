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


def generate_kuramoto(
    M: int,
    T: int,
    dt: float = 0.002,
    K: float = 1.5,
    k: int = 1,
    w: float = 1.0,
    omega_mean: float = 2 * np.pi * 0.1,
    omega_std: float = 0.01,
    eta: float = 0.0,
    transients: int = 2000,
    output: str = "sin",
    directed: bool = False,
    rng=None,
) -> np.ndarray:
    rng = _resolve_rng(None, rng)
    A = np.zeros((M, M))
    if directed:
        for i in range(M):
            A[i, (i + 1) % M] = w
    else:
        for i in range(M):
            for d in range(1, k + 1):
                A[i, (i + d) % M] = w
                A[i, (i - d) % M] = w
    theta = rng.uniform(0, 2 * np.pi, size=M)
    omega = rng.normal(omega_mean, omega_std, size=M)
    steps = transients + T
    Y = np.zeros((steps, M))
    for t in range(steps):
        if output == "sin":
            Y[t] = np.sin(theta)
        elif output == "cos":
            Y[t] = np.cos(theta)
        else:
            Y[t] = theta
        S = np.sin(theta[None, :] - theta[:, None])
        coupling_term = K * (A * S).sum(axis=1)
        noise = eta * np.sqrt(dt) * rng.normal(size=M)
        theta = np.mod(theta + (omega + coupling_term) * dt + noise, 2 * np.pi)
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

