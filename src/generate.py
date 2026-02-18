from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
from numpy.random import default_rng

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
    topology: str = "ring-symmetric",  # <--- NEW PARAMETER
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
    right = np.roll(I, 1, axis=1)   # i influenced by i+1 (flow from right)
    left = np.roll(I, -1, axis=1)   # i influenced by i-1 (flow from left)
    if topology == "ring-symmetric":
        neighbors = left + right
    elif topology == "ring-unidirectional":
        neighbors = left 
    elif topology == "all-to-all":
        neighbors = np.ones((M, M)) - I  # full coupling except self
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
    """
    Standard Exponential distribution (rate parameter gamma = 1).
    """
    rng = _resolve_rng(None, rng)
    # NumPy uses scale parameter beta = 1/gamma.
    # For gamma = 1, scale = 1.0.
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
    Discretised with Euler–Maruyama in log space:
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
    # NEW:
    vel_std: float = 0.2,        # strength of initial velocity (0.0 reproduces old behavior)
    vel_modes: int = 6,          # bandlimit (smoothness). <= M//2
    vel_decay: float = 2.0,      # amplitudes ~ 1/k^vel_decay
) -> np.ndarray:
    """
    Simulates the 1D Wave Equation: d^2z/dt^2 = c^2 * d^2z/du^2

    Periodic BC.
    Initial displacement: Gaussian.
    Initial velocity: random smooth (bandlimited Fourier) if vel_std>0.
    """
    rng = _resolve_rng(seed, rng)

    dx = 1.0 / M
    dt = 0.2 * dx / c
    coeff = (c * dt / dx) ** 2

    # --- Initial displacement z(x,0) ---
    coords = np.arange(M, dtype=float)
    center = M / 2.0
    sigma = M / 20.0

    z_prev = np.exp(-((coords - center) ** 2) / (2.0 * sigma**2))
    z_prev = z_prev / np.max(np.abs(z_prev))

    # --- NEW: Initial velocity v(x,0) = dz/dt ---
    v0 = np.zeros(M, dtype=float)
    if vel_std > 0:
        x = coords / M  # [0,1)
        K = int(min(max(1, vel_modes), M // 2))
        ks = np.arange(1, K + 1, dtype=float)

        phases = rng.uniform(0.0, 2.0 * np.pi, size=K)
        amps = rng.normal(0.0, 1.0, size=K) / (ks ** vel_decay)

        # bandlimited random field
        for k, a, ph in zip(ks, amps, phases):
            v0 += a * np.sin(2.0 * np.pi * k * x + ph)

        v0 -= v0.mean()  # remove mean to avoid global drift
        s = np.std(v0)
        if s > 0:
            v0 = (v0 / s) * vel_std

    # --- First time step z(x,dt) ---
    lap_prev = _laplacian_1d(z_prev)
    z_curr = z_prev + dt * v0 + 0.5 * coeff * lap_prev

    # --- Integrate ---
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
    "ring-symmetric": "bidirectional-list",
    "symmetric": "bidirectional-list",
    "grid-four": "grid-four",
    "grid_four": "grid-four",
    "grid-4": "grid-four",
    "grid": "grid-four",
    "ring-unidirectional": "ring-unidirectional",
    "unidirectional": "ring-unidirectional",
    "directed-ring": "ring-unidirectional",
    "splay": "ring-unidirectional",
}

def _normalize_connectivity(name: str) -> str:
    key = name.strip().lower().replace(" ", "-")
    key = key.replace("_", "-")
    if key not in _KURAMOTO_CONN_ALIASES:
        raise ValueError(
            f"Unknown connectivity '{name}'. "
            "Expected one of all-to-all, bidirectional-list/ring(-symmetric), grid-four, ring-unidirectional."
        )
    return _KURAMOTO_CONN_ALIASES[key]


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
    if connectivity == "ring-unidirectional":
        A = np.zeros((M, M), float)
        for i in range(M):
            A[i, (i - 1) % M] = 1.0  # i influenced by i-1 only
        return A
    raise ValueError(f"Unsupported connectivity '{connectivity}'.")


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

def generate_kuramoto(
    M: int,
    T: int,
    dt: float = 0.002,  # NOTE: user code had 0.002, ensuring high res
    K: float = 1.5,
    k_ring: int = 1,
    omega_mean: float = 2 * np.pi * 0.1,
    omega_std: float = 0.01,
    noise_std: float | None = None,
    eta: float = 0.0,
    transients: int = 2000,
    output: str = "sin",
    coupling_scheme: str = "bidirectional_list",
    *,
    connectivity: str | None = None,
    topology: str | None = None,
    k: float | None = None,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    
    rng = _resolve_rng(None, rng)
    
    # ... (Parameter validation remains the same) ...
    
    conn_name = connectivity or topology or coupling_scheme
    conn_canonical = _normalize_connectivity(conn_name)
    _ensure_grid_compatible(conn_canonical, M)

    # RESOLVE COUPLING STRENGTH
    coupling = float(k if k is not None else K)
    # Allow legacy noise_std param as alias for eta
    resolved_eta = float(eta if noise_std is None else noise_std)
    
    # PARAMETER SETUP
    base_frequency = float(omega_mean)
    
    # DECISION LOGIC: USE PYTHON OR PYCLUSTERING?
    # We force Python solver for 'ring-unidirectional' because pyclustering 
    # lacks a native enum for it, and we want precise control over the wave physics.
    # Always use Python solver; ring-unidirectional is not supported by pyclustering.
    data = _simulate_python_kuramoto(
        M=M,
        T=T,
        dt=dt,
        coupling=coupling,
        connectivity=conn_canonical,
        k_ring=k_ring,
        omega_mean=base_frequency,
        omega_std=omega_std,
        eta=resolved_eta,
        transients=transients,
        output=output,
        rng=rng,
        zscore=zscore,
    )
    return data

def generate_kuramoto_all_to_all(*args, k: float, **kwargs) -> np.ndarray:
    return generate_kuramoto(*args, k=k, connectivity="all-to-all", **kwargs)


def generate_kuramoto_bidirectional_list(*args, k: float, **kwargs) -> np.ndarray:
    return generate_kuramoto(*args, k=k, connectivity="bidirectional-list", **kwargs)


def generate_kuramoto_grid_four(*args, k: float, **kwargs) -> np.ndarray:
    return generate_kuramoto(*args, k=k, connectivity="grid-four", **kwargs)


def generate_mackey_glass(
    M: int,
    T: int,
    tau: float = 17.0,      # Physical delay
    beta: float = 0.2,      # Feedback strength
    gamma: float = 0.1,     # Decay
    n: int = 10,            # Nonlinearity power
    coupling: float = 0.05, # Diffusive coupling strength
    transients: int = 1000,
    dt: float = 0.1,        # High-Fidelity Integration Step
    topology: str = "ring-unidirectional",
    rng=None,
    zscore: bool = True,
):
    """
    Generates M coupled Mackey-Glass oscillators.
    High-fidelity DDE simulation (dt=0.1) with no downsampling.
    
    NOTE: With dt=0.1 and tau=17.0, the lag occurs at index 170.
    Ensure T > 680 so that 170 is within the T/4 xcorr scan window.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. Calculate Lag in Steps
    tau_steps = int(round(tau / dt))
    if tau_steps < 1:
        raise ValueError(f"tau ({tau}) must be >= dt ({dt}).")

    # 2. Allocation
    steps = transients + T
    # Buffer: History (tau) + Simulation (steps) + 1 safety
    X = np.zeros((steps + tau_steps + 1, M))

    # 3. Initialization (History)
    X[:tau_steps + 1] = rng.uniform(0.5, 1.5, size=(tau_steps + 1, M))

    # 4. Topology Neighbors
    neighbors_left = None
    neighbors_right = None
    
    if topology == "ring-unidirectional":
        neighbors_left = np.roll(np.arange(M), 1)
    elif topology == "ring-symmetric":
        neighbors_left = np.roll(np.arange(M), 1)
        neighbors_right = np.roll(np.arange(M), -1)

    # 5. Integration Loop
    start_k = tau_steps
    end_k = start_k + steps
    
    for k in range(start_k, end_k):
        curr_state = X[k]
        delayed_state = X[k - tau_steps]
        
        # A. Internal Dynamics
        interaction = (beta * delayed_state) / (1.0 + delayed_state**n)
        decay = -gamma * curr_state
        
        # B. External Coupling
        coupling_force = 0.0
        if topology == "ring-unidirectional":
            neighbor = X[k, neighbors_left]
            # Standard Diffusive: coupling * (neighbor - self)
            coupling_force = coupling * (neighbor - curr_state)
        elif topology == "ring-symmetric":
            left = X[k, neighbors_left]
            right = X[k, neighbors_right]
            # Standard Laplacian: coupling * sum(neighbors - self)
            coupling_force = coupling * ((left - curr_state) + (right - curr_state))

        # C. Euler Step
        dxdt = interaction + decay + coupling_force
        X[k + 1] = curr_state + dxdt * dt

    # 6. Output Slicing
    output = X[start_k + transients : start_k + transients + T]

    if zscore:
        mus = output.mean(axis=0)
        sigs = output.std(axis=0)
        sigs[sigs < 1e-6] = 1.0
        output = (output - mus) / sigs

    return output


def _softplus(x: np.ndarray) -> np.ndarray:
    # stable log(1+exp(x))
    return np.logaddexp(0.0, x)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Numerically-stable logistic sigmoid.
    """
    x = np.asarray(x)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _E_softplus_gaussian(sigma2: float | np.ndarray, *, n_gh: int = 40) -> np.ndarray:
    """
    Approx E[softplus(Z)] for Z ~ N(0, sigma2) using Gauss–Hermite quadrature.
    Vectorized over sigma2.
    """
    s2 = np.asarray(sigma2, dtype=float)
    # GH nodes/weights for ∫ e^{-x^2} f(x) dx
    x, w = np.polynomial.hermite.hermgauss(n_gh)  # shapes (n_gh,)

    # E[f(Z)] = 1/sqrt(pi) * sum_i w_i f( sqrt(2*s2)*x_i )
    scale = np.sqrt(2.0 * s2)[..., None]  # (...,1)
    vals = _softplus(scale * x[None, :])  # (..., n_gh)
    return (vals @ w) / np.sqrt(np.pi)    # (...,)


def _resolve_g(
    g: str | Callable[[np.ndarray], np.ndarray],
    *,
    sigma2: float | np.ndarray | None,
    params: dict[str, float] | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Resolve coupling/mixing function g from a string key or callable."""
    if callable(g):
        return g

    name = str(g).lower().strip()
    p: dict[str, float] = {} if params is None else dict(params)

    if name in {"id", "identity", "linear"}:
        return lambda z: z
    if name in {"affine", "linear_affine"}:
        if "alpha" not in p:
            raise ValueError("g='affine' requires g_params with key 'alpha' (slope).")
        alpha = float(p["alpha"])
        beta = float(p.get("beta", 0.0))  # intercept (note: zscore=True will remove offsets)
        return lambda z, _a=alpha, _b=beta: _a * z + _b
    if name == "tanh":
        return np.tanh
    if name == "sin":
        return np.sin
    if name in {"sigmoid", "logistic"}:
        if "beta" not in p:
            raise ValueError("g='sigmoid' requires g_params with key 'beta' (slope).")
        beta = float(p["beta"])
        bias = float(p.get("bias", 0.0))  # optional shift: sigmoid(beta*z + bias)
        return lambda z, _k=beta, _b=bias: _sigmoid(_k * z + _b)
    if name in {"abs", "absolute"}:
        return np.abs
    if name in {"square", "pow2", "quadratic"}:
        return lambda z: z**2
    if name in {"square_centered", "pow2_centered", "quadratic_centered"}:
        if sigma2 is None:
            raise ValueError("g='square_centered' requires sigma2 (Var of the latent).")
        s2 = np.asarray(sigma2, dtype=float)
        return lambda z, _s2=s2: z**2 - _s2

    # --- NEW: skewed monotone nonlinearities ---
    if name == "exp":
        # clip to avoid inf; tweak bounds if you want more/less skew
        return lambda z: np.exp(np.clip(z, -50.0, 50.0))

    if name in {"exp_centered", "exp_shifted"}:
        # centered: exp(z) - E[exp(Z)] where Z~N(0,s2), E = exp(s2/2)
        if sigma2 is None:
            raise ValueError("g='exp_centered' requires sigma2 (Var of the latent).")
        s2 = np.asarray(sigma2, dtype=float)
        mu = np.exp(0.5 * s2)
        return lambda z, _mu=mu: np.exp(np.clip(z, -50.0, 50.0)) - _mu

    if name == "softplus":
        return _softplus

    if name in {"softplus_centered", "softplus_shifted"}:
        if sigma2 is None:
            raise ValueError("g='softplus_centered' requires sigma2 (Var of the latent).")
        s2 = np.asarray(sigma2, dtype=float)
        mu = _E_softplus_gaussian(s2, n_gh=40)
        return lambda z, _mu=mu: _softplus(z) - _mu

    raise ValueError(
        f"Unknown g='{g}'. Supported: identity, affine, sigmoid, tanh, sin, abs, square, "
        f"square_centered, exp, exp_centered, softplus, softplus_centered (or pass a callable)."
    )


def generate_case_i(
    M: int,
    T: int,
    *,
    a: float | np.ndarray = 0.6,
    b: float | np.ndarray = 0.25,
    c: float | np.ndarray = 0.0,
    g: str | Callable[[np.ndarray], np.ndarray] = "identity",
    g_params: dict[str, float] | None = None,
    interaction: str = "prev",
    topology: str = "ring",
    boundary_value: float = 0.0,
    noise_std: float | np.ndarray = 0.1,
    transients: int = 0,
    x0: float | np.ndarray | None = 0.0,
    init_std: float = 1.0,
    sigma2: float | None = None,
    target_rho: float | None = None,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    Generic ring/chain-coupled M-channel MTS with configurable nonlinear coupling.

    Update (per channel i):
        x_{t+1}^{(i)} = a*x_t^{(i)} + b*x_t^{(i-1)} + c*g( h(x_t^{(i)}, x_t^{(i-1)}) ) + eps_{t+1}^{(i)}

    target_rho:
      - If provided, rescale the *linear* backbone (a,b) to have spectral radius <= target_rho,
        in the same way as generate_varma.
      - Only allowed when c == 0 (purely linear system). No normalization is done for nonlinear cases.

    Returns array of shape (T, M).
    """
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if transients < 0:
        raise ValueError(f"transients must be >= 0, got {transients}")
    if target_rho is not None and target_rho <= 0:
        raise ValueError(f"target_rho must be positive, got {target_rho}")

    rng = _resolve_rng(None, rng)

    a_arr = np.broadcast_to(np.asarray(a, dtype=float), (M,)).copy()
    b_arr = np.broadcast_to(np.asarray(b, dtype=float), (M,)).copy()
    c_arr = np.broadcast_to(np.asarray(c, dtype=float), (M,))
    noise_std_arr = np.broadcast_to(np.asarray(noise_std, dtype=float), (M,))

    # --- Spectral radius normalization (linear case only) ---
    if target_rho is not None:
        if not np.allclose(c_arr, 0.0):
            raise ValueError("target_rho is only supported for the linear case (requires c == 0).")

        topology_key = str(topology).lower().strip()
        A = np.zeros((M, M), dtype=float)
        idx = np.arange(M)
        A[idx, idx] = a_arr

        if topology_key == "ring":
            prev_cols = (idx - 1) % M
            A[idx, prev_cols] = b_arr
        elif topology_key == "chain":
            if M > 1:
                A[idx[1:], idx[:-1]] = b_arr[1:]  # row i uses b_i * x_{i-1}
            # note: boundary_value is an external constant, not part of A
        else:
            raise ValueError("topology must be 'ring' or 'chain'")

        ev = np.linalg.eigvals(A)
        sr = float(np.max(np.abs(ev))) if ev.size else 0.0
        if sr >= float(target_rho) and sr > 0.0:
            scale = float(target_rho) / sr
            a_arr *= scale
            b_arr *= scale
    # -------------------------------------------------------

    g_fn = _resolve_g(g, sigma2=sigma2, params=g_params)

    total_T = T + transients
    paths = np.zeros((total_T, M), dtype=float)

    if x0 is None:
        paths[0] = rng.normal(scale=float(init_std), size=(M,))
    else:
        paths[0] = np.broadcast_to(np.asarray(x0, dtype=float), (M,))

    noise = rng.normal(size=(max(total_T - 1, 0), M)) * noise_std_arr

    interaction_key = str(interaction).lower().strip()
    topology_key = str(topology).lower().strip()

    for t in range(1, total_T):
        cur = paths[t - 1]

        if topology_key == "ring":
            prev = np.roll(cur, 1)
        elif topology_key == "chain":
            prev = np.empty_like(cur)
            prev[0] = float(boundary_value)
            prev[1:] = cur[:-1]
        else:
            raise ValueError("topology must be 'ring' or 'chain'")

        if interaction_key == "prev":
            z = prev
        elif interaction_key == "product":
            z = cur * prev
        elif interaction_key == "sum":
            z = cur + prev
        elif interaction_key == "diff":
            z = cur - prev
        else:
            raise ValueError("interaction must be one of: prev, product, sum, diff")

        paths[t] = a_arr * cur + b_arr * prev + np.broadcast_to(c_arr, (M,)) * g_fn(z) + noise[t - 1]

    out = paths[transients:] if transients > 0 else paths
    return _maybe_zscore(out, zscore=zscore)


def generate_case_ii(
    M: int,
    T: int,
    *,
    a: float | np.ndarray = 0.6,
    b: float | np.ndarray = 0.25,
    g: str | Callable[[np.ndarray], np.ndarray] = "identity",
    topology: str = "ring",
    boundary_value: float = 0.0,
    noise_std: float | np.ndarray = 0.1,
    transients: int = 0,
    z0: float | np.ndarray | None = 0.0,
    init_std: float = 1.0,
    sigma2: float | None = None,
    target_rho: float | None = None,
    rng=None,
    zscore: bool = True,
    return_latents: bool = False,
):
    """
    Case II: latent independent AR(1) channels + instantaneous (lag-0) mixing.

    Latents (independent across channels):
        z_{t+1}^{(i)} = a_i * z_t^{(i)} + eps_{t+1}^{(i)},  eps ~ N(0, noise_std_i^2)

    Observation / mixing layer (instantaneous ring/chain):
        x_t^{(i)} = z_t^{(i)} + b_i * g( z_t^{(i-1)} )

    - topology='ring': z_t^{(i-1)} uses wraparound (i-1 mod M)
    - topology='chain': z_t^{(i-1)} uses boundary_value for i=0

    target_rho:
      - If provided, rescales the AR coefficients a_i so that max_i |a_i| <= target_rho.
        (This is the analogue of "stability control" for independent AR(1) latents.)

    Returns:
      - x of shape (T, M) by default
      - (x, z) if return_latents=True
    """
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if transients < 0:
        raise ValueError(f"transients must be >= 0, got {transients}")
    if target_rho is not None and target_rho <= 0:
        raise ValueError(f"target_rho must be positive, got {target_rho}")

    # Keep same repo hygiene: use your RNG resolver + zscore helper.
    rng = _resolve_rng(None, rng)  # noqa: F821 (expected to exist in your repo)

    a_arr = np.broadcast_to(np.asarray(a, dtype=float), (M,)).copy()
    b_arr = np.broadcast_to(np.asarray(b, dtype=float), (M,)).copy()
    noise_std_arr = np.broadcast_to(np.asarray(noise_std, dtype=float), (M,)).copy()

    # inside generate_case_ii, after a_arr and noise_std_arr are defined:
    sigma2_auto = None
    g_name = str(g).lower().strip() if not callable(g) else ""
    if sigma2 is None and g_name.endswith(("centered", "shifted")):
        # per-channel stationary variance for AR(1) latents
        denom = 1.0 - a_arr**2
        if np.any(denom <= 0):
            raise ValueError("Cannot auto-compute sigma2 when |a|>=1.")
        sigma2_auto = (noise_std_arr**2) / denom

    g_fn = _resolve_g(g, sigma2=sigma2 if sigma2 is not None else sigma2_auto)

    # --- Stability control for independent AR(1) latents ---
    if target_rho is not None:
        maxabs = float(np.max(np.abs(a_arr))) if a_arr.size else 0.0
        if maxabs > float(target_rho) and maxabs > 0.0:
            a_arr *= float(target_rho) / maxabs
    # ------------------------------------------------------

    total_T = T + transients
    z = np.zeros((total_T, M), dtype=float)

    # Initial condition for latents
    if z0 is None:
        z[0] = rng.normal(scale=float(init_std), size=(M,))
    else:
        z[0] = np.broadcast_to(np.asarray(z0, dtype=float), (M,))

    # Latent noise
    eps = rng.normal(size=(max(total_T - 1, 0), M)) * noise_std_arr

    # Generate latents (independent across channels)
    for t in range(1, total_T):
        z[t] = a_arr * z[t - 1] + eps[t - 1]

    # Instantaneous mixing at each t (lag-0 across channels)
    topology_key = str(topology).lower().strip()
    if topology_key == "ring":
        prev = np.roll(z, 1, axis=1)
    elif topology_key == "chain":
        prev = np.empty_like(z)
        prev[:, 0] = float(boundary_value)
        prev[:, 1:] = z[:, :-1]
    else:
        raise ValueError("topology must be 'ring' or 'chain'")

    x = z + g_fn(prev) * b_arr  # b_arr broadcasts over time

    # Drop transients then z-score (same pattern as generate_case_i)
    x_out = x[transients:] if transients > 0 else x
    z_out = z[transients:] if transients > 0 else z

    x_out = _maybe_zscore(x_out, zscore=zscore)  # noqa: F821 (expected to exist in your repo)
    if return_latents:
        z_out = _maybe_zscore(z_out, zscore=zscore)
        return x_out, z_out

    return x_out


def generate_case_iii(
    M: int,
    T: int,
    *,
    mode: str = "lagged_drive_pairs",
    # ---- lagged_drive_pairs params ----
    a_driver: float | np.ndarray = 0.0,   # set 0.0 for i.i.d. drivers -> MI(lag0) ~ 0
    a_resp: float | np.ndarray = 0.6,
    coupling: float | np.ndarray = 0.8,   # driver -> responder (lag-1)
    noise_std_driver: float | np.ndarray = 1.0,
    noise_std_resp: float | np.ndarray = 1.0,
    # ---- instantaneous_common_cause params ----
    a_base: float | np.ndarray = 0.6,     # independent AR(1) base per channel
    noise_std_base: float | np.ndarray = 1.0,
    latent_std: float = 1.0,              # Z_t ~ N(0, latent_std^2), i.i.d. over t
    alpha: float | np.ndarray = 1.0,      # mixing strength per channel
    # ---- shared ----
    transients: int = 0,
    x0: float | np.ndarray | None = 0.0,
    init_std: float = 1.0,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    Case III: two regimes to separate lag-0 MI from directed TE.

    mode="lagged_drive_pairs":
        Channels are paired (0->1, 2->3, ...). Even channels are "drivers",
        odd channels are "responders".

        driver:   x_{t+1}^{(2k)}   = a_driver * x_t^{(2k)} + eps^D
        responder x_{t+1}^{(2k+1)} = a_resp   * x_t^{(2k+1)} + coupling * x_t^{(2k)} + eps^R

        If a_driver=0 (i.i.d. drivers), then x_t^{(2k)} is independent of x_t^{(2k+1)}
        (which depends only on past driver), so MI at lag-0 is ~0 while TE (2k -> 2k+1) is high.

    mode="instantaneous_common_cause":
        Independent AR(1) bases + instantaneous i.i.d. common cause:
            u_{t+1}^{(i)} = a_base * u_t^{(i)} + eta^{(i)}
            x_t^{(i)}     = u_t^{(i)} + alpha_i * Z_t,  Z_t i.i.d.

        This creates large lag-0 MI across channels (shared Z_t) but TE ~ 0 (no cross-lag mechanism).

    Returns array shape (T, M).
    """
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if transients < 0:
        raise ValueError(f"transients must be >= 0, got {transients}")

    rng = _resolve_rng(None, rng)  # expected to exist in your repo

    total_T = T + transients
    x = np.zeros((total_T, M), dtype=float)

    if x0 is None:
        x[0] = rng.normal(scale=float(init_std), size=(M,))
    else:
        x[0] = np.broadcast_to(np.asarray(x0, dtype=float), (M,))

    mode_key = str(mode).lower().strip()

    if mode_key == "lagged_drive_pairs":
        aD = np.broadcast_to(np.asarray(a_driver, dtype=float), (M,))
        aR = np.broadcast_to(np.asarray(a_resp, dtype=float), (M,))
        c = np.broadcast_to(np.asarray(coupling, dtype=float), (M,))
        sD = np.broadcast_to(np.asarray(noise_std_driver, dtype=float), (M,))
        sR = np.broadcast_to(np.asarray(noise_std_resp, dtype=float), (M,))

        drivers = (np.arange(M) % 2 == 0)  # even indices
        responders = ~drivers               # odd indices

        eps = rng.normal(size=(max(total_T - 1, 0), M))
        for t in range(1, total_T):
            prev = x[t - 1]

            # driver update (no cross-channel terms)
            nxt = np.empty_like(prev)
            nxt[drivers] = aD[drivers] * prev[drivers] + eps[t - 1, drivers] * sD[drivers]

            # responder update: depends on *paired* driver at lag-1
            # For odd i, paired driver is i-1
            paired_driver = np.zeros_like(prev)
            if M > 1:
                paired_driver[responders] = prev[np.where(responders)[0] - 1]
            nxt[responders] = (
                aR[responders] * prev[responders]
                + c[responders] * paired_driver[responders]
                + eps[t - 1, responders] * sR[responders]
            )

            x[t] = nxt

    elif mode_key == "instantaneous_common_cause":
        a = np.broadcast_to(np.asarray(a_base, dtype=float), (M,))
        s = np.broadcast_to(np.asarray(noise_std_base, dtype=float), (M,))
        alpha_arr = np.broadcast_to(np.asarray(alpha, dtype=float), (M,))

        # simulate independent base u_t in-place in x, then add Z_t each time
        eps = rng.normal(size=(max(total_T - 1, 0), M))
        for t in range(1, total_T):
            x[t] = a * x[t - 1] + eps[t - 1] * s

        Z = rng.normal(scale=float(latent_std), size=(total_T,))
        x = x + Z[:, None] * alpha_arr[None, :]

    else:
        raise ValueError("mode must be one of: 'lagged_drive_pairs', 'instantaneous_common_cause'")

    out = x[transients:] if transients > 0 else x
    return _maybe_zscore(out, zscore=zscore)  # expected to exist in your repo


def generate_mts_master(
    M: int,
    T: int,
    *,
    g: str = "sigmoid",
    beta: float = 1.0,
    a: float = 1.0,
    noise_std: float = 1.0,
    alpha: float = 0.0,
    transients: int = 0,
    seed: int | None = None,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    Generate M-channel MTS by applying a shared nonlinear filter to a common AR(1) master signal.

    Model:
        Master signal:  z_{t+1} = a * z_t + ε_t,  where ε_t ~ N(0, noise_std²)
        Observations:   X_t^{(i)} = g(z_t; β) + α * η_t^{(i)},  where η ~ N(0, 1)

    All channels share the same filter g with the same β, differing only in
    independent observational noise (controlled by α).

    Parameters
    ----------
    M : int
        Number of channels.
    T : int
        Length of the output time series.
    g : str
        Filter function: 'sigmoid', 'bell', or 'affine'.
        - sigmoid: 1 / (1 + exp(-β * z))   (increasing)
        - bell:    exp(-β * z²)            (Gaussian bump centered at 0)
        - affine:  β * z                   (linear scaling)
    beta : float
        Filter parameter (same for all channels).
    a : float
        AR(1) persistence coefficient for the master signal. Default 1.0 (random walk).
    noise_std : float
        Standard deviation of AR(1) innovations. Default 1.0.
    alpha : float
        Observational noise level. Set to 0 to disable. Default 0.0.
    transients : int
        Number of initial time steps to discard (burn-in). Default 0.
    seed : int | None
        Random seed.
    rng : Generator | None
        NumPy random generator (takes precedence over seed).
    zscore : bool
        Whether to z-score each channel. Default True.

    Returns
    -------
    np.ndarray
        Array of shape (T, M).
    """
    if M <= 0:
        raise ValueError("M must be positive")
    if T <= 0:
        raise ValueError("T must be positive")

    rng = _resolve_rng(seed, rng)

    # --- Generate master AR(1) signal ---
    total_T = T + transients
    z = np.zeros(total_T, dtype=float)
    eps = rng.normal(0, noise_std, size=total_T)
    for t in range(1, total_T):
        z[t] = a * z[t - 1] + eps[t]

    # --- Define filter functions ---
    g_name = str(g).lower().strip()
    if g_name == "sigmoid":
        def g_fn(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-beta * x))
    elif g_name == "bell":
        def g_fn(x: np.ndarray) -> np.ndarray:
            return np.exp(-beta * x ** 2)
    elif g_name == "affine":
        def g_fn(x: np.ndarray) -> np.ndarray:
            return beta * x
    else:
        raise ValueError(f"Unknown g='{g}'. Supported: 'sigmoid', 'bell', 'affine'.")

    # --- Apply filter to master signal ---
    z_use = z[transients:]  # shape (T,)
    filtered = g_fn(z_use)  # shape (T,)

    # --- Broadcast to M channels and add observational noise ---
    X = np.tile(filtered[:, None], (1, M))  # shape (T, M)
    if alpha > 0:
        eta = rng.normal(0, 1, size=(T, M))
        X = X + alpha * eta

    return _maybe_zscore(X, zscore=zscore)


GENERATOR_REGISTRY: Dict[str, GeneratorFn] = {
    "varma": generate_varma,
    "var": generate_varma,
    "varma_shuffled": generate_varma_shuffled,
    "cml_logistic": generate_cml_logistic,
    "gbm": generate_gbm,
    "geometric_brownian_motion": generate_gbm,
    "mackey_glass": generate_mackey_glass,
    "kuramoto": generate_kuramoto,
    "kuramoto_all_to_all": generate_kuramoto_all_to_all,
    "kuramoto_bidirectional_list": generate_kuramoto_bidirectional_list,
    "kuramoto_grid_four": generate_kuramoto_grid_four,
    "gaussian_noise": generate_gaussian_noise,
    "cauchy_noise": generate_cauchy_noise,
    "exponential_noise": generate_exponential_noise,
    "wave_1d": generate_wave_1d,
    "wave_2d": generate_wave_2d,
    "case_i": generate_case_i,
    "case_ii": generate_case_ii,
    "case_iii": generate_case_iii,
    "mts_master": generate_mts_master,
}


def available_generators() -> list[str]:
    return sorted(GENERATOR_REGISTRY.keys())


def generate_series(name: str, *, seed: int | None = None, **params: Any) -> np.ndarray:
    if name not in GENERATOR_REGISTRY:
        raise KeyError(f"Unknown generator '{name}'. Known: {available_generators()}")
    rng = _resolve_rng(seed)
    gen = GENERATOR_REGISTRY[name]
    return gen(rng=rng, **params)
