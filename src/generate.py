from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
from numpy.random import default_rng

try:
    from pyclustering.nnet.sync import (
        conn_represent,
        conn_type,
        initial_type,
        solve_type,
        sync_network,
    )
except ImportError as exc:  # pragma: no cover
    sync_network = None  # type: ignore[assignment]
    conn_type = conn_represent = initial_type = solve_type = None  # type: ignore[assignment]
    _PYCLUSTERING_IMPORT_ERROR = exc
else:
    _PYCLUSTERING_IMPORT_ERROR = None

GeneratorFn = Callable[..., np.ndarray]


def _global_rng():
    return default_rng(123456789)


def _resolve_rng(seed: int | None, rng=None):
    if rng is not None:
        return rng
    if seed is None:
        return _global_rng()
    return default_rng(seed)


def _zscore_channels(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, ddof=0, keepdims=True)
    zero_mask = std < 1e-12
    std = np.where(zero_mask, 1.0, std)
    normalised = (data - mean) / std
    if np.any(zero_mask):
        cols = zero_mask.reshape(-1)
        eps = 1e-6
        normalised[0, cols] = eps
        if normalised.shape[0] > 1:
            normalised[1, cols] = -eps
    return normalised


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
    return _zscore_channels(X[transients:])


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
    respect_transients: bool = False,
    rng=None,
) -> np.ndarray:
    rng = _resolve_rng(None, rng)
    M = int(M)
    T = int(T)
    # Match reference generator by default (no burn-in). Enable `respect_transients`
    # when an experiment explicitly wants to discard initial samples.
    transient_samples = int(transients) if (respect_transients and transients > 0) else 0

    def logistic(x, a):
        return 1 - a * x**2

    def iterate_map(x, epsilon, f):
        fx = f(x)
        left = np.roll(fx, 1)
        right = np.roll(fx, -1)
        return (1 - epsilon) * fx + (epsilon / 2.0) * (left + right)

    lattice_M = max(M, 20)
    total_samples = max(1, transient_samples + T)
    baseline_samples = max(total_samples, 1000)
    states = np.zeros((baseline_samples, lattice_M), dtype=float)
    states[0] = rng.random(lattice_M)
    f = lambda x: logistic(x, alpha)
    for t in range(1, baseline_samples):
        states[t] = iterate_map(states[t - 1], eps, f)
    if baseline_samples < total_samples:
        raise ValueError(
            f"Insufficient CML samples (need {total_samples}, have {baseline_samples})."
        )
    offset = (lattice_M - M) // 2
    if offset < 0 or offset + M > lattice_M:
        raise ValueError(
            f"Cannot crop {M} channels from lattice size {lattice_M}."
        )
    cropped = states[:total_samples, offset : offset + M]
    usable = cropped[transient_samples : transient_samples + T]
    return _zscore_channels(usable)


def _laplacian_1d(z: np.ndarray) -> np.ndarray:
    return np.roll(z, -1) - 2.0 * z + np.roll(z, 1)


def _laplacian_2d(z: np.ndarray) -> np.ndarray:
    return (
        np.roll(z, -1, axis=0)
        - 2.0 * z
        + np.roll(z, 1, axis=0)
        + np.roll(z, -1, axis=1)
        - 2.0 * z
        + np.roll(z, 1, axis=1)
    )


def generate_wave_1d(
    M: int,
    T: int,
    *,
    c: float = 10.0,
    seed: int | None = None,
    rng=None,
) -> np.ndarray:
    rng = np.random.default_rng(seed) if seed is not None else _resolve_rng(None, rng)
    dx = 1.0 / M
    dt = 0.2 * dx / c
    coeff = (c * dt / dx) ** 2
    coords = np.arange(M, dtype=float)
    sigma = M / 20.0
    z_prev = np.exp(-((coords - M / 2.0) ** 2) / (2.0 * sigma**2))
    z_prev = z_prev / np.max(np.abs(z_prev))
    lap_prev = _laplacian_1d(z_prev)
    z_curr = z_prev + 0.5 * coeff * lap_prev
    samples = np.zeros((T, M), dtype=float)
    samples[0] = z_prev
    if T > 1:
        samples[1] = z_curr
    for t in range(2, T):
        lap = _laplacian_1d(z_curr)
        z_next = 2.0 * z_curr - z_prev + coeff * lap
        samples[t] = z_next
        z_prev, z_curr = z_curr, z_next
    return _zscore_channels(samples)


def generate_wave_2d(
    M: int,
    T: int,
    *,
    c: float = 10.0,
    seed: int | None = None,
    rng=None,
) -> np.ndarray:
    side = int(np.sqrt(M))
    if side * side != M:
        raise ValueError("Wave 2D generator requires M to be a perfect square.")
    rng = np.random.default_rng(seed) if seed is not None else _resolve_rng(None, rng)
    dx = 1.0 / side
    dt = 0.2 * dx / c
    coeff = (c * dt / dx) ** 2
    x = np.arange(side, dtype=float)
    X, Y = np.meshgrid(x, x, indexing="ij")
    sigma = M / 20.0
    z_prev = np.exp(-(((X - side / 2.0) ** 2 + (Y - side / 2.0) ** 2) / (2.0 * sigma**2)))
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
    return _zscore_channels(samples)


_KURAMOTO_CONN_ALIASES = {
    "all-to-all": "all-to-all",
    "all_to_all": "all-to-all",
    "alltoall": "all-to-all",
    "fully_connected": "all-to-all",
    "full": "all-to-all",
    "bidirectional-list": "bidirectional-list",
    "bidirectional_list": "bidirectional-list",
    "list": "bidirectional-list",
    "ring": "bidirectional-list",
    "grid-four": "grid-four",
    "grid_four": "grid-four",
    "grid-4": "grid-four",
    "grid": "grid-four",
}


def _require_pyclustering() -> None:
    if _PYCLUSTERING_IMPORT_ERROR is not None:
        raise ImportError(
            "pyclustering>=0.10.1 is required for Kuramoto generators."
        ) from _PYCLUSTERING_IMPORT_ERROR


def _normalize_connectivity(name: str) -> str:
    key = name.strip().lower().replace(" ", "-")
    key = key.replace("_", "-")
    if key not in _KURAMOTO_CONN_ALIASES:
        raise ValueError(
            f"Unknown connectivity '{name}'. "
            "Expected one of all-to-all, bidirectional-list, grid-four."
        )
    return _KURAMOTO_CONN_ALIASES[key]


def _conn_type_from_name(name: str) -> conn_type:
    mapping = {
        "all-to-all": conn_type.ALL_TO_ALL,
        "bidirectional-list": conn_type.LIST_BIDIR,
        "grid-four": conn_type.GRID_FOUR,
    }
    return mapping[name]


def _ensure_grid_compatible(connectivity: str, M: int) -> None:
    if connectivity != "grid-four":
        return
    side = int(np.sqrt(M))
    if side * side != M:
        raise ValueError(
            f"grid-four connectivity requires M to be a perfect square (got M={M})."
        )


def _build_kuramoto_adjacency(
    M: int,
    connectivity: str,
    k_ring: int,
) -> np.ndarray:
    if connectivity == "all-to-all":
        A = np.ones((M, M), float)
        np.fill_diagonal(A, 0.0)
        return A
    if connectivity == "bidirectional-list":
        A = np.zeros((M, M), float)
        for i in range(M):
            for d in range(1, k_ring + 1):
                A[i, (i + d) % M] = 1.0
                A[i, (i - d) % M] = 1.0
        return A
    if connectivity == "grid-four":
        side = int(np.sqrt(M))
        A = np.zeros((M, M), float)
        for idx in range(M):
            r, c = divmod(idx, side)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = (r + dr) % side
                nc = (c + dc) % side
                j = nr * side + nc
                A[idx, j] = 1.0
        return A
    raise ValueError(f"Unsupported connectivity '{connectivity}'.")


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
    *,
    connectivity: str | None = None,
    k: float | None = None,
    rng=None,
) -> np.ndarray:
    """
    Kuramoto network driven by pyclustering's sync_network.

    The legacy parameters `k_ring` and `omega_std` are retained for compatibility:
    `k_ring` must remain 1 (pyclustering's LIST_BIDIR couples only two neighbours)
    and `omega_std` is treated as a small global perturbation of omega_mean.
    """
    _require_pyclustering()
    rng = _resolve_rng(None, rng)
    if k_ring != 1:
        raise ValueError(
            "pyclustering back-end does not support k_ring != 1 for bidirectional lists."
        )
    coupling = float(k if k is not None else K)
    if not np.isfinite(coupling):
        raise ValueError("Coupling strength must be finite.")
    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be positive.")
    total_steps = int(transients + T)
    if total_steps <= 0:
        raise ValueError("Total number of steps must be positive.")
    sim_time = max(float(total_steps) * dt, dt)

    conn_name = connectivity or coupling_scheme
    if not isinstance(conn_name, str):
        raise ValueError("connectivity must be provided as a string.")
    conn_canonical = _normalize_connectivity(conn_name)
    _ensure_grid_compatible(conn_canonical, M)
    conn = _conn_type_from_name(conn_canonical)

    base_frequency = float(omega_mean)
    frequency = base_frequency
    if omega_std:
        frequency += float(rng.normal(scale=omega_std))

    data = _simulate_pyclustering_kuramoto(
        M=M,
        T=T,
        dt=dt,
        coupling=coupling,
        omega_mean=frequency,
        omega_std=omega_std,
        eta=eta,
        transients=transients,
        output=output,
        conn=conn,
        sim_time=sim_time,
        total_steps=total_steps,
        rng=rng,
    )
    if data is None:
        print(
            "[WARN] pyclustering Kuramoto simulation produced non-finite values; "
            "falling back to Python integrator."
        )
        data = _simulate_python_kuramoto(
            M=M,
            T=T,
            dt=dt,
            coupling=coupling,
            connectivity=conn_canonical,
            k_ring=k_ring,
            omega_mean=base_frequency,
            omega_std=omega_std,
            eta=eta,
            transients=transients,
            output=output,
            rng=rng,
        )
    return data


def _simulate_pyclustering_kuramoto(
    *,
    M: int,
    T: int,
    dt: float,
    coupling: float,
    omega_mean: float,
    omega_std: float,
    eta: float,
    transients: int,
    output: str,
    conn: conn_type,
    sim_time: float,
    total_steps: int,
    rng,
) -> np.ndarray | None:
    net = sync_network(
        num_osc=M,
        weight=coupling,
        frequency=omega_mean,
        type_conn=conn,
        representation=conn_represent.MATRIX,
        initial_phases=initial_type.RANDOM_GAUSSIAN,
        ccore=True,
    )
    dynamic = net.simulate_static(
        steps=total_steps,
        time=sim_time,
        solution=solve_type.RK4,
        collect_dynamic=True,
    )
    phase = np.asarray(dynamic.output, dtype=float)
    if not np.isfinite(phase).all():
        return None
    if phase.ndim != 2 or phase.shape[1] != M:
        raise ValueError(f"Unexpected phase matrix shape {phase.shape}.")
    if phase.shape[0] == total_steps + 1:
        phase = phase[1:]
    if phase.shape[0] < total_steps:
        raise ValueError(
            f"Insufficient samples from pyclustering dynamic ({phase.shape[0]} < {total_steps})."
        )
    usable_phase = phase[-total_steps:]
    usable_phase = usable_phase[transients:, :]
    if usable_phase.shape[0] < T:
        raise ValueError(
            f"Need at least {T} samples after transients, got {usable_phase.shape[0]}."
        )
    usable_phase = usable_phase[:T, :]
    if eta:
        noise = rng.normal(scale=np.sqrt(dt) * eta, size=usable_phase.shape)
        usable_phase = usable_phase + noise
    if output == "sin":
        data = np.sin(usable_phase)
    elif output == "cos":
        data = np.cos(usable_phase)
    elif output == "phase":
        data = usable_phase
    else:
        raise ValueError(f"Unknown output '{output}'.")
    if not np.isfinite(data).all():
        return None
    return _zscore_channels(data)


def _simulate_python_kuramoto(
    *,
    M: int,
    T: int,
    dt: float,
    coupling: float,
    connectivity: str,
    k_ring: int,
    omega_mean: float,
    omega_std: float,
    eta: float,
    transients: int,
    output: str,
    rng,
) -> np.ndarray:
    A = _build_kuramoto_adjacency(M, connectivity, k_ring)
    degree = A.sum(axis=1)
    inv_degree = np.where(degree > 0, 1.0 / degree, 0.0)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=M)
    omega = rng.normal(omega_mean, omega_std, size=M)
    steps = transients + T
    Y = np.zeros((steps, M), float)
    for t in range(steps):
        if output == "sin":
            Y[t] = np.sin(theta)
        elif output == "cos":
            Y[t] = np.cos(theta)
        else:
            Y[t] = theta
        phase_diff = theta[None, :] - theta[:, None]
        coupling_term = (A * np.sin(phase_diff)).sum(axis=1)
        dtheta = omega + coupling * inv_degree * coupling_term
        noise = eta * np.sqrt(dt) * rng.normal(size=M)
        theta = np.mod(theta + dtheta * dt + noise, 2.0 * np.pi)
    return _zscore_channels(Y[transients:])


def generate_kuramoto_all_to_all(*args, k: float, **kwargs) -> np.ndarray:
    return generate_kuramoto(*args, k=k, connectivity="all-to-all", **kwargs)


def generate_kuramoto_bidirectional_list(*args, k: float, **kwargs) -> np.ndarray:
    return generate_kuramoto(*args, k=k, connectivity="bidirectional-list", **kwargs)


def generate_kuramoto_grid_four(*args, k: float, **kwargs) -> np.ndarray:
    return generate_kuramoto(*args, k=k, connectivity="grid-four", **kwargs)


GENERATOR_REGISTRY: Dict[str, GeneratorFn] = {
    "var": generate_var,
    "cml_logistic": generate_cml_logistic,
    "kuramoto": generate_kuramoto,
    "kuramoto_all_to_all": generate_kuramoto_all_to_all,
    "kuramoto_bidirectional_list": generate_kuramoto_bidirectional_list,
    "kuramoto_grid_four": generate_kuramoto_grid_four,
    "gaussian_noise": generate_gaussian_noise,
    "cauchy_noise": generate_cauchy_noise,
    "wave_1d": generate_wave_1d,
    "wave_2d": generate_wave_2d,
}


def available_generators() -> list[str]:
    return sorted(GENERATOR_REGISTRY.keys())


def generate_series(name: str, *, seed: int | None = None, **params: Any) -> np.ndarray:
    if name not in GENERATOR_REGISTRY:
        raise KeyError(f"Unknown generator '{name}'. Known: {available_generators()}")
    rng = _resolve_rng(seed)
    gen = GENERATOR_REGISTRY[name]
    return gen(rng=rng, **params)
