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


import numpy as np
from numpy.random import default_rng

# ... (Include your existing _global_rng, _resolve_rng, _zscore_channels, _maybe_zscore helpers here) ...

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


def generate_kuramoto(
    M: int,
    T: int,
    dt: float = 0.01, # Standard dt
    K: float = 1.0,   # Coupling strength
    topology: str = "all-to-all", # 'all-to-all', 'ring-symmetric', 'ring-unidirectional'
    omega_mean: float = 1.0,
    omega_std: float = 0.1,
    noise_std: float = 0.05, # Dynamic noise (eta)
    transients: int = 1000,
    rng=None,
    zscore: bool = True,
):
    """
    Highly optimized Kuramoto generator.
    Removes pyclustering dependency.
    """
    if rng is None: rng = np.random.default_rng()
    
    # 1. Initialization
    steps = transients + T
    # Intrinsic frequencies
    omega = rng.normal(loc=omega_mean, scale=omega_std, size=M)
    # Initial phases
    theta = rng.uniform(0, 2*np.pi, M)
    
    # Pre-allocate output (only storing the post-transient part to save RAM if T is large)
    # We store the *Sine* of the phase, as that is the observable time series.
    X = np.zeros((T, M), dtype=np.float32)

    # 2. Pre-calculation for noise
    # We add sqrt(dt) scaling to noise standard deviation for Euler-Maruyama
    noise_scale = noise_std * np.sqrt(dt)

    # 3. Simulation Loop
    for t in range(-transients, T):
        
        # --- TOPOLOGY SWITCHING ---
        
        # A. All-to-All (Mean Field Optimization) - O(M)
        if topology == "all-to-all":
            # Order parameter Z = R * e^(i*Psi) = (1/M) * sum(e^(i*theta))
            Z = np.mean(np.exp(1j * theta))
            R = np.abs(Z)
            Psi = np.angle(Z)
            # Interaction = K * R * sin(Psi - theta)
            interaction = K * R * np.sin(Psi - theta)
            
        # B. Unidirectional Ring (Splay/Wave) - O(M)
        elif topology == "ring-unidirectional":
            # i depends on i-1. Flow is Right -> Left in array index.
            theta_prev = np.roll(theta, 1) 
            interaction = K * np.sin(theta_prev - theta)
            
        # C. Symmetric Ring (Diffusive) - O(M)
        elif topology == "ring-symmetric":
            theta_left = np.roll(theta, 1)  # i-1
            theta_right = np.roll(theta, -1) # i+1
            # Average of neighbors
            interaction = (K/2) * (np.sin(theta_left - theta) + np.sin(theta_right - theta))
            
        else:
            raise ValueError(f"Unknown topology: {topology}")
            
        # --- UPDATE STEP (Euler-Maruyama) ---
        theta += (omega + interaction) * dt + rng.normal(scale=noise_scale, size=M)
        
        # Store after transients
        if t >= 0:
            X[t] = np.sin(theta)

    # 4. Return Z-Scored
    if zscore:
        # Simple safe z-score
        mus = X.mean(axis=0)
        sigs = X.std(axis=0)
        sigs[sigs < 1e-6] = 1.0
        X = (X - mus) / sigs
        
    return X

# 3. UPDATE GENERATOR TO BYPASS PYCLUSTERING FOR UNIDIRECTIONAL

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

import numpy as np

import numpy as np

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


GENERATOR_REGISTRY: Dict[str, GeneratorFn] = {
    "varma": generate_varma,
    "var": generate_varma,
    "varma_shuffled": generate_varma_shuffled,
    "cml_logistic": generate_cml_logistic,
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
}


def available_generators() -> list[str]:
    return sorted(GENERATOR_REGISTRY.keys())


def generate_series(name: str, *, seed: int | None = None, **params: Any) -> np.ndarray:
    if name not in GENERATOR_REGISTRY:
        raise KeyError(f"Unknown generator '{name}'. Known: {available_generators()}")
    rng = _resolve_rng(seed)
    gen = GENERATOR_REGISTRY[name]
    return gen(rng=rng, **params)
