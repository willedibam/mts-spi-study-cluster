from __future__ import annotations

import numpy as np

from ._common import _maybe_zscore, _resolve_rng


def _latent_driver(T: int, ar_coeff: float, rng: np.random.Generator) -> np.ndarray:
    """Mean-reverting AR(1) latent driver of length T."""
    z = np.zeros(T)
    z[0] = rng.normal()
    for t in range(1, T):
        z[t] = ar_coeff * z[t - 1] + rng.normal()
    return z


def _noisy_copy(signal: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return signal + rng.normal(0, sigma, size=signal.shape)


def generate_topology_uniform(
    M: int,
    T: int,
    ar_coeff: float = 0.95,
    sigma: float = 0.1,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    Uniform topology: all M channels are independent noisy copies of a
    single latent AR(1) driver. No preferential coupling structure.
    Returns shape (T, M).
    """
    rng = _resolve_rng(None, rng)
    z = _latent_driver(T, ar_coeff, rng)
    X = np.empty((T, M))
    for i in range(M):
        X[:, i] = _noisy_copy(z, sigma, rng)
    return _maybe_zscore(X, zscore=zscore)


def generate_topology_hub_spoke(
    M: int,
    T: int,
    ar_coeff: float = 0.95,
    sigma: float = 0.1,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    Hub-spoke (star) topology: channel 0 copies the latent driver,
    all others copy channel 0. Pairs involving ch0 have strong coupling;
    leaf-leaf pairs have weaker (indirect) coupling.
    Returns shape (T, M).
    """
    rng = _resolve_rng(None, rng)
    z = _latent_driver(T, ar_coeff, rng)
    X = np.empty((T, M))
    X[:, 0] = _noisy_copy(z, sigma, rng)
    for i in range(1, M):
        X[:, i] = _noisy_copy(X[:, 0], sigma, rng)
    return _maybe_zscore(X, zscore=zscore)


def generate_topology_chain(
    M: int,
    T: int,
    ar_coeff: float = 0.95,
    sigma: float = 0.1,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    Chain (sequential) topology: channel 0 copies the driver,
    channel i copies channel i-1. Coupling degrades with distance.
    Returns shape (T, M).
    """
    rng = _resolve_rng(None, rng)
    z = _latent_driver(T, ar_coeff, rng)
    X = np.empty((T, M))
    X[:, 0] = _noisy_copy(z, sigma, rng)
    for i in range(1, M):
        X[:, i] = _noisy_copy(X[:, i - 1], sigma, rng)
    return _maybe_zscore(X, zscore=zscore)


def generate_topology_clustered(
    M: int,
    T: int,
    ar_coeff: float = 0.95,
    sigma: float = 0.1,
    inter_cluster_noise: float = 0.5,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    Clustered (two-community) topology: the latent driver spawns two
    sub-drivers with additional noise. First M//2 channels copy sub-driver A,
    remaining copy sub-driver B. Strong intra-cluster, weak inter-cluster coupling.
    Returns shape (T, M).
    """
    rng = _resolve_rng(None, rng)
    z = _latent_driver(T, ar_coeff, rng)
    z_a = z + rng.normal(0, inter_cluster_noise, size=T)
    z_b = z + rng.normal(0, inter_cluster_noise, size=T)

    M_a = M // 2
    X = np.empty((T, M))
    for i in range(M_a):
        X[:, i] = _noisy_copy(z_a, sigma, rng)
    for i in range(M_a, M):
        X[:, i] = _noisy_copy(z_b, sigma, rng)
    return _maybe_zscore(X, zscore=zscore)


# =============================================================================
# VAR(1)-based topology generators (degree-controlled adjacency)
#
# Design: all classes share the same generative mechanism —
#   X_t = A X_{t-1} + eps_t
# where A = phi*I + coupling*G/rho(G), and G is a prescribed adjacency
# matrix.  By using 2-regular graphs (ring, two disjoint 8-cycles,
# four disjoint 4-cycles) plus a 4x4 lattice, we control for:
#   - same coupling weight per edge (within the 2-regular group)
#   - same spectral-radius budget
# The *only* distinguishing feature is the wiring pattern.
# Each sample additionally receives a random node permutation so
# that positional information (node index) carries no signal.
# =============================================================================


def _build_var_matrix(
    G: np.ndarray, phi: float, coupling: float, target_rho: float,
) -> np.ndarray:
    """Build A = phi*I + coupling*G, scaled so spectral_radius(A) <= target_rho."""
    M = G.shape[0]
    A = phi * np.eye(M) + coupling * G
    sr = np.max(np.abs(np.linalg.eigvals(A)))
    if sr > 1e-12 and sr >= target_rho:
        A *= target_rho / sr
    return A


def _var1_simulate(
    A: np.ndarray, T: int, noise_std: float, transients: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate X_t = A X_{t-1} + eps_t, discard initial transients."""
    M = A.shape[0]
    steps = transients + T
    X = np.zeros((steps, M))
    for t in range(1, steps):
        X[t] = A @ X[t - 1] + rng.normal(0, noise_std, size=M)
    return X[transients:]


def _permute_channels(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly permute channel (column) indices."""
    perm = rng.permutation(X.shape[1])
    return X[:, perm]


# ── adjacency builders ──────────────────────────────────────────────


def _adj_cycle(M: int) -> np.ndarray:
    """Single Hamiltonian cycle on M nodes."""
    G = np.zeros((M, M))
    for i in range(M):
        j = (i + 1) % M
        G[i, j] = 1.0
        G[j, i] = 1.0
    return G


def _adj_multi_cycle(M: int, num_components: int) -> np.ndarray:
    """Union of `num_components` disjoint equal-length cycles."""
    size = M // num_components
    if size * num_components != M:
        raise ValueError(
            f"M={M} not evenly divisible by num_components={num_components}"
        )
    if size < 3:
        raise ValueError(
            f"Component size {size} too small for a cycle (min 3)"
        )
    G = np.zeros((M, M))
    for c in range(num_components):
        offset = c * size
        for i in range(size):
            a = offset + i
            b = offset + (i + 1) % size
            G[a, b] = 1.0
            G[b, a] = 1.0
    return G


def _adj_lattice(M: int) -> np.ndarray:
    """4-connected grid (requires M to be a perfect square)."""
    side = int(np.round(np.sqrt(M)))
    if side * side != M:
        raise ValueError(f"M={M} must be a perfect square for lattice topology")
    G = np.zeros((M, M))
    for r in range(side):
        for c in range(side):
            node = r * side + c
            if c + 1 < side:
                G[node, node + 1] = 1.0
                G[node + 1, node] = 1.0
            if r + 1 < side:
                G[node, node + side] = 1.0
                G[node + side, node] = 1.0
    return G


# ── public generators ───────────────────────────────────────────────


def generate_var_ring(
    M: int,
    T: int,
    phi: float = 0.5,
    coupling: float = 0.3,
    noise_std: float = 1.0,
    target_rho: float = 0.95,
    transients: int = 200,
    permute: bool = True,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    VAR(1) on a single M-cycle (ring).
    All nodes degree 2, one connected component.
    """
    rng = _resolve_rng(None, rng)
    G = _adj_cycle(M)
    A = _build_var_matrix(G, phi, coupling, target_rho)
    X = _var1_simulate(A, T, noise_std, transients, rng)
    if permute:
        X = _permute_channels(X, rng)
    return _maybe_zscore(X, zscore=zscore)


def generate_var_tworing(
    M: int,
    T: int,
    phi: float = 0.5,
    coupling: float = 0.3,
    noise_std: float = 1.0,
    target_rho: float = 0.95,
    transients: int = 200,
    permute: bool = True,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    VAR(1) on two disjoint M/2-cycles.
    All nodes degree 2, two connected components.
    """
    rng = _resolve_rng(None, rng)
    G = _adj_multi_cycle(M, num_components=2)
    A = _build_var_matrix(G, phi, coupling, target_rho)
    X = _var1_simulate(A, T, noise_std, transients, rng)
    if permute:
        X = _permute_channels(X, rng)
    return _maybe_zscore(X, zscore=zscore)


def generate_var_fourring(
    M: int,
    T: int,
    phi: float = 0.5,
    coupling: float = 0.3,
    noise_std: float = 1.0,
    target_rho: float = 0.95,
    transients: int = 200,
    permute: bool = True,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    VAR(1) on four disjoint M/4-cycles.
    All nodes degree 2, four connected components.
    """
    rng = _resolve_rng(None, rng)
    G = _adj_multi_cycle(M, num_components=4)
    A = _build_var_matrix(G, phi, coupling, target_rho)
    X = _var1_simulate(A, T, noise_std, transients, rng)
    if permute:
        X = _permute_channels(X, rng)
    return _maybe_zscore(X, zscore=zscore)


def generate_var_lattice(
    M: int,
    T: int,
    phi: float = 0.5,
    coupling: float = 0.3,
    noise_std: float = 1.0,
    target_rho: float = 0.95,
    transients: int = 200,
    permute: bool = True,
    rng=None,
    zscore: bool = True,
) -> np.ndarray:
    """
    VAR(1) on a sqrt(M) x sqrt(M) grid (4-connected lattice).
    Degree varies (2–4). Denser than the cycle classes, but spectral-
    radius normalisation keeps the effective coupling comparable.
    """
    rng = _resolve_rng(None, rng)
    G = _adj_lattice(M)
    A = _build_var_matrix(G, phi, coupling, target_rho)
    X = _var1_simulate(A, T, noise_std, transients, rng)
    if permute:
        X = _permute_channels(X, rng)
    return _maybe_zscore(X, zscore=zscore)
