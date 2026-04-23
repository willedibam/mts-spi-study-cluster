from __future__ import annotations

import numpy as np

from ._common import _maybe_zscore, _resolve_rng


def generate_cml_logistic(
    M: int,
    T: int,
    alpha: float = 1.7522,
    eps: float = 0.00115,
    transients: int = 100,
    sample_every: int = 1,
    rng=None,
    zscore: bool = True,
    init_state: np.ndarray | None = None,
    return_final_state: bool = False,
):
    """
    Coupled map lattice of logistic maps with diffusive ring coupling.

    sample_every: retain every `sample_every`-th step after the burn-in
        (`observe every delta points`). sample_every=1 keeps every step.
        Increasing it decorrelates adjacent samples; useful when the fast
        chaotic timescale would swamp the slower spatial coupling signal.

    init_state: optional lattice state to seed the simulation with (shape
        (max(M, 100),)). When provided the random initialisation is skipped;
        used to chain runs across parameter values (quasi-static sweep).
    return_final_state: when True, return (data, final_lattice_state) so the
        caller can feed it back as init_state for the next run.
    """
    rng = _resolve_rng(None, rng)
    M = int(M)
    T = int(T)
    transients = max(0, int(transients))
    sample_every = max(1, int(sample_every))

    def logistic(x, a):
        return 1 - a * x**2

    def iterate_map(x, epsilon, f):
        fx = f(x)
        left = np.roll(fx, 1)
        right = np.roll(fx, -1)
        return (1 - epsilon) * fx + (epsilon / 2.0) * (left + right)

    lattice_M = max(M, 100)
    offset = (lattice_M - M) // 2
    total_steps = transients + T * sample_every
    states = np.zeros((total_steps, lattice_M), dtype=float)
    if init_state is None:
        states[0] = rng.random(lattice_M)
    else:
        init = np.asarray(init_state, dtype=float)
        if init.shape != (lattice_M,):
            raise ValueError(
                f"init_state must have shape ({lattice_M},), got {init.shape}"
            )
        states[0] = init
    f = lambda x: logistic(x, alpha)
    for t in range(1, total_steps):
        states[t] = iterate_map(states[t - 1], eps, f)
    cropped = states[transients:, offset : offset + M]
    usable = cropped[::sample_every][:T]
    result = _maybe_zscore(usable, zscore=zscore)
    if return_final_state:
        return result, states[-1].copy()
    return result


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
            A[i, (i - 1) % M] = 1.0
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
    dt: float = 0.002,
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

    conn_name = connectivity or topology or coupling_scheme
    conn_canonical = _normalize_connectivity(conn_name)
    _ensure_grid_compatible(conn_canonical, M)

    coupling = float(k if k is not None else K)
    resolved_eta = float(eta if noise_std is None else noise_std)
    base_frequency = float(omega_mean)

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
    tau: float = 17.0,
    beta: float = 0.2,
    gamma: float = 0.1,
    n: int = 10,
    coupling: float = 0.05,
    transients: int = 1000,
    dt: float = 0.1,
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

    tau_steps = int(round(tau / dt))
    if tau_steps < 1:
        raise ValueError(f"tau ({tau}) must be >= dt ({dt}).")

    steps = transients + T
    X = np.zeros((steps + tau_steps + 1, M))
    X[:tau_steps + 1] = rng.uniform(0.5, 1.5, size=(tau_steps + 1, M))

    neighbors_left = None
    neighbors_right = None

    if topology == "ring-unidirectional":
        neighbors_left = np.roll(np.arange(M), 1)
    elif topology == "ring-symmetric":
        neighbors_left = np.roll(np.arange(M), 1)
        neighbors_right = np.roll(np.arange(M), -1)

    start_k = tau_steps
    end_k = start_k + steps

    for k in range(start_k, end_k):
        curr_state = X[k]
        delayed_state = X[k - tau_steps]

        interaction = (beta * delayed_state) / (1.0 + delayed_state**n)
        decay = -gamma * curr_state

        coupling_force = 0.0
        if topology == "ring-unidirectional":
            neighbor = X[k, neighbors_left]
            coupling_force = coupling * (neighbor - curr_state)
        elif topology == "ring-symmetric":
            left = X[k, neighbors_left]
            right = X[k, neighbors_right]
            coupling_force = coupling * ((left - curr_state) + (right - curr_state))

        dxdt = interaction + decay + coupling_force
        X[k + 1] = curr_state + dxdt * dt

    output = X[start_k + transients : start_k + transients + T]

    if zscore:
        mus = output.mean(axis=0)
        sigs = output.std(axis=0)
        sigs[sigs < 1e-6] = 1.0
        output = (output - mus) / sigs

    return output
