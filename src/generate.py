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


def _maybe_zscore(data: np.ndarray, *, zscore: bool = True) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    return _zscore_channels(arr) if zscore else arr


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
    rng=None,
    zscore: bool = True,
):
    """
    Generate a multivariate VARMA(1,1) process on a ring topology.

    Model:
        X_t = A X_{t-1} + ε_t + B ε_{t-1},
        ε_t ~ N(0, noise_std^2 I_M)

    Topology:
        - A and B are built on a *ring*:
            * diagonal: self terms
            * off-diagonal: nearest neighbours (i-1, i+1 mod M)

    Parameters
    ----------
    M : int
        Number of channels (nodes) in the multivariate time series.
    T : int
        Number of time steps to return (after discarding transients).
    phi : float
        Autoregressive self-correlation (diagonal of A).
        Larger -> stronger persistence of each channel's own past.
    coupling : float
        Autoregressive ring coupling strength (off-diagonals of A).
        Larger -> stronger influence of nearest neighbours.
    ma_phi : float
        Moving-average self term (diagonal of B).
        Set to 0.0 to obtain a pure VAR(1) (no MA component).
    ma_coupling : float
        Moving-average ring coupling term (off-diagonals of B).
        Set to 0.0 to obtain a pure VAR(1).
    noise_std : float
        Standard deviation of the innovation noise ε_t (Gaussian).
    transients : int
        Number of initial steps to simulate and discard as burn-in.
    target_rho : float
        Target spectral radius for the AR matrix A. If ρ(A) ≥ target_rho,
        A is rescaled as A ← (target_rho / ρ(A)) A.
    rng :
        Optional np.random.Generator or seed; resolved via _resolve_rng.

    Returns
    -------
    X : (T, M) np.ndarray
        Multivariate time series (z-scored when `zscore=True`) after discarding transients.
    """
    rng = _resolve_rng(None, rng)
    I = np.eye(M)
    right = np.roll(I, 1, axis=1)   # neighbour (i+1 mod M)
    left = np.roll(I, -1, axis=1)   # neighbour (i-1 mod M)
    A = phi * I + coupling * (left + right)     # Autoregressive matrix A (ring topology)
    ev = np.linalg.eigvals(A)
    sr = np.max(np.abs(ev))
    if sr >= target_rho:
        A = A * (target_rho / sr)
    B = ma_phi * I + ma_coupling * (left + right)     # Moving-average matrix B (ring topology)
    steps = transients + T
    X = np.zeros((steps, M), float)
    eps = rng.normal(0.0, noise_std, size=(steps, M))
    for t in range(1, steps):
        X[t] = A @ X[t - 1] + eps[t] + B @ eps[t - 1]
    return _maybe_zscore(X[transients:], zscore=zscore)


def generate_varma_shuffled(
    M: int,
    T: int,
    # Pass through all standard VARMA parameters
    phi: float = 0.6,
    coupling: float = 0.4,
    ma_phi: float = 0.2,
    ma_coupling: float = 0.1,
    noise_std: float = 0.1,
    transients: int = 100,
    target_rho: float = 0.99,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    rng = _resolve_rng(None, rng)
    X = generate_varma(
        M=M, T=T, 
        phi=phi, coupling=coupling, 
        ma_phi=ma_phi, ma_coupling=ma_coupling,
        noise_std=noise_std, transients=transients, 
        target_rho=target_rho, rng=rng, zscore=False
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
    """
    Standard Exponential distribution (rate parameter gamma = 1).
    """
    rng = _resolve_rng(None, rng)
    # NumPy uses scale parameter beta = 1/gamma.
    # For gamma = 1, scale = 1.0.
    data = rng.exponential(scale=1.0, size=(T, M))
    return _maybe_zscore(data, zscore=zscore) if zscore else data


def generate_cml_logistic(
    M: int,
    T: int,
    alpha: float = 1.7522,
    eps: float = 0.00115,
    delta: int = 12,
    transients: int = 100,
    respect_transients: bool = False,
    rng=None,
    zscore: bool = True,
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
    return _maybe_zscore(usable, zscore=zscore)


def _laplacian_1d(z: np.ndarray) -> np.ndarray:
    """
    Finite difference Laplacian in 1D with periodic boundary conditions.
    Approximates d^2z/du^2.
    """
    return np.roll(z, -1) - 2.0 * z + np.roll(z, 1)


def _laplacian_2d(z: np.ndarray) -> np.ndarray:
    """
    Finite difference Laplacian in 2D with periodic boundary conditions.
    Approximates d^2z/du^2 + d^2z/dv^2.
    """
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
) -> np.ndarray:
    """
    Simulates the 1D Wave Equation: d^2z/dt^2 = c^2 * d^2z/du^2
    
    Context parameters:
      - c = 10 (default)
      - Periodic boundary conditions
      - Initial condition: Gaussian with sigma = M/20
    """
    rng = _resolve_rng(seed, rng)
    
    # 1. Setup Space and Time (Unit domain)
    dx = 1.0 / M
    # Courant stability condition: c * dt/dx <= 1. Use 0.2 safety factor.
    dt = 0.2 * dx / c
    coeff = (c * dt / dx) ** 2

    # 2. Initial Conditions (Gaussian)
    coords = np.arange(M, dtype=float)
    center = M / 2.0
    sigma = M / 20.0
    
    z_prev = np.exp(-((coords - center) ** 2) / (2.0 * sigma**2))
    z_prev = z_prev / np.max(np.abs(z_prev))

    # 3. First Time Step (t=1)
    # Assume initial velocity dz/dt = 0; Taylor expansion
    lap_prev = _laplacian_1d(z_prev)
    z_curr = z_prev + 0.5 * coeff * lap_prev

    # 4. Integration Loop
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
    
    Context parameters:
      - c = 10 (default)
      - Periodic boundary conditions
      - Initial condition: Gaussian with sigma = M/20
      - M is the total number of processes (nodes). The grid side is sqrt(M).
    """
    rng = _resolve_rng(seed, rng)

    # 1. Setup Grid
    side = int(np.sqrt(M))
    if side * side != M:
        raise ValueError(f"Wave 2D generator requires M to be a perfect square. Got M={M}.")
    
    dx = 1.0 / side
    dt = 0.2 * dx / c
    coeff = (c * dt / dx) ** 2

    # 2. Initial Conditions (2D Gaussian)
    x = np.arange(side, dtype=float)
    X, Y = np.meshgrid(x, x, indexing="ij")
    center = side / 2.0
    sigma = M / 20.0
    
    dist_sq = (X - center) ** 2 + (Y - center) ** 2
    z_prev = np.exp(-(dist_sq) / (2.0 * sigma**2))
    z_prev = z_prev / np.max(np.abs(z_prev))

    # 3. First Time Step (t=1)
    lap_prev = _laplacian_2d(z_prev)
    z_curr = z_prev + 0.5 * coeff * lap_prev

    # 4. Integration Loop
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
    zscore: bool = True,
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
        zscore=zscore,
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
            zscore=zscore,
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
    zscore: bool,
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
    return _maybe_zscore(data, zscore=zscore)


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
    zscore: bool,
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
    return _maybe_zscore(Y[transients:], zscore=zscore)


def generate_kuramoto_all_to_all(*args, k: float, **kwargs) -> np.ndarray:
    return generate_kuramoto(*args, k=k, connectivity="all-to-all", **kwargs)


def generate_kuramoto_bidirectional_list(*args, k: float, **kwargs) -> np.ndarray:
    return generate_kuramoto(*args, k=k, connectivity="bidirectional-list", **kwargs)


def generate_kuramoto_grid_four(*args, k: float, **kwargs) -> np.ndarray:
    return generate_kuramoto(*args, k=k, connectivity="grid-four", **kwargs)


GENERATOR_REGISTRY: Dict[str, GeneratorFn] = {
    "varma": generate_varma,
    "var": generate_varma,
    "varma_shuffled": generate_varma_shuffled,
    "cml_logistic": generate_cml_logistic,
    "kuramoto": generate_kuramoto,
    "kuramoto_all_to_all": generate_kuramoto_all_to_all,
    "kuramoto_bidirectional_list": generate_kuramoto_bidirectional_list,
    "kuramoto_grid_four": generate_kuramoto_grid_four,
    "gaussian_noise": generate_gaussian_noise,
    "cauchy_noise": generate_cauchy_noise,
    "exponential_noise": generate_exponential_noise,
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
