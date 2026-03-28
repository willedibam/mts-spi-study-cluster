from __future__ import annotations

import numpy as np

from ._common import _maybe_zscore, _resolve_rng


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


# ---------------------------------------------------------------------------
# Kuramoto Chat A: directed 3-node phase-coupled motifs
# ---------------------------------------------------------------------------

_KURAMOTO_CHAT_A_CLASSES = {
    0: "chain",     # 0→1→2
    1: "fork",      # 0→1, 0→2
    2: "collider",  # 1→0, 2→0
}


def _kuramoto_chat_a_motif_edges(motif_class: int) -> list[tuple[int, int]]:
    """Return directed edges for the 3-node Kuramoto motif."""
    if motif_class == 0:
        return [(0, 1), (1, 2)]
    if motif_class == 1:
        return [(0, 1), (0, 2)]
    if motif_class == 2:
        return [(1, 0), (2, 0)]
    raise ValueError(f"Unknown motif_class {motif_class}; expected 0, 1, or 2")


def generate_kuramoto_chat_a(
    M: int,
    T: int,
    motif_class: int = 0,
    dt: float = 0.05,
    K_lo: float = 3.0,
    K_hi: float = 6.0,
    omega_mean: float = 2 * np.pi * 1.0,
    omega_std: float = 0.3,
    eta: float = 0.3,
    transients: int = 500,
    output: str = "sin",
    rng=None,
    zscore: bool = True,
    return_internals: bool = False,
) -> np.ndarray | tuple:
    """
    Kuramoto Chat A: embed a directed 3-node phase-coupled motif into M oscillators.

    Motif nodes are coupled via a bidirectional adjacency derived from the
    directed motif edges (since phase coupling is symmetric, both A→B and B←A
    create a sin(θ_B - θ_A) term). The remaining M-3 oscillators are
    uncoupled (each runs at its own natural frequency + noise).

    Classes:
        0 = chain   (0→1→2) → coupled pairs {0-1, 1-2}
        1 = fork    (0→1, 0→2) → coupled pairs {0-1, 0-2}
        2 = collider (1→0, 2→0) → coupled pairs {0-1, 0-2}

    Note: chain and collider produce different directed edges but the
    *same* undirected coupling skeleton {0-1, 1-2} vs {0-1, 0-2}.
    Chain has skeleton {0-1, 1-2}; fork and collider share {0-1, 0-2}.
    The asymmetric SPI statistics (e.g. directed information transfer)
    should still distinguish these via the direction of influence.

    Coupling K drawn from Uniform(K_lo, K_hi) per sample.
    Returns shape (T, M), or (data, ChatMotifInternals) if return_internals=True.
    """
    from .chat import ChatMotifInternals, _permute_and_merge

    if M < 3:
        raise ValueError(f"M must be >= 3 for a 3-node motif, got {M}")
    rng = _resolve_rng(None, rng)

    edges = _kuramoto_chat_a_motif_edges(motif_class)
    K = float(rng.uniform(K_lo, K_hi))

    # Build 3x3 symmetric adjacency from directed edges
    A_motif = np.zeros((3, 3))
    for src, dst in edges:
        A_motif[src, dst] = 1.0
        A_motif[dst, src] = 1.0  # phase coupling is bidirectional

    degree = A_motif.sum(axis=1)
    inv_degree = np.where(degree > 0, 1.0 / degree, 0.0)

    # Natural frequencies for ALL M oscillators
    omega = rng.normal(omega_mean, omega_std, size=M)

    # --- Simulate motif nodes (3 coupled oscillators) ---
    theta_motif = rng.uniform(0.0, 2 * np.pi, size=3)
    steps = transients + T
    Y_motif = np.zeros((steps, 3))
    for t in range(steps):
        Y_motif[t] = np.sin(theta_motif) if output == "sin" else np.cos(theta_motif)
        phase_diff = theta_motif[None, :] - theta_motif[:, None]
        coupling_term = (A_motif * np.sin(phase_diff)).sum(axis=1)
        dtheta = omega[:3] + K * inv_degree * coupling_term
        noise = eta * np.sqrt(dt) * rng.normal(size=3)
        theta_motif = np.mod(theta_motif + dtheta * dt + noise, 2 * np.pi)
    X_motif = Y_motif[transients:]

    # --- Simulate nuisance nodes (M-3 uncoupled oscillators) ---
    n_nuis = M - 3
    if n_nuis > 0:
        theta_nuis = rng.uniform(0.0, 2 * np.pi, size=n_nuis)
        Y_nuis = np.zeros((steps, n_nuis))
        for t in range(steps):
            Y_nuis[t] = np.sin(theta_nuis) if output == "sin" else np.cos(theta_nuis)
            dtheta_n = omega[3:]
            noise_n = eta * np.sqrt(dt) * rng.normal(size=n_nuis)
            theta_nuis = np.mod(theta_nuis + dtheta_n * dt + noise_n, 2 * np.pi)
        X_nuis = Y_nuis[transients:]
    else:
        X_nuis = np.empty((T, 0))

    # Permute and merge
    data, motif_indices, edges_permuted = _permute_and_merge(
        X_motif, X_nuis, edges, rng
    )
    data = _maybe_zscore(data, zscore=zscore)

    if return_internals:
        internals = ChatMotifInternals(
            motif_node_indices=motif_indices,
            motif_edges=edges_permuted,
            class_label=_KURAMOTO_CHAT_A_CLASSES[motif_class],
            coupling_values={"K": K},
        )
        return data, internals
    return data


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
