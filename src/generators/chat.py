"""
Generators for the EEML GNN-on-SPI study.

Three generator families, each producing (T, M) MTS with ground-truth
motif edge annotations stored as auxiliary returns.

Generator A  (var_chat_a): directed 3-node VAR motifs (chain / fork / collider)
Generator B  (var_chat_b): common-driver confounder (no direct 2→3 vs direct 2→3)
Generator C  (var_chat_c): nonlinear coupling (linear vs tanh)
Generator D  (var_chat_d): lag discrimination (lag 1 vs lag 3)

All generators embed a small motif into M nodes; remaining nodes are
independent AR(1) nuisance processes. Motif node indices are randomly
permuted so positional information carries no signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._common import _maybe_zscore, _resolve_rng


# ---------------------------------------------------------------------------
# Internals dataclass — returned alongside the MTS when return_internals=True
# ---------------------------------------------------------------------------


@dataclass
class ChatMotifInternals:
    """Ground-truth motif information for explanation evaluation."""

    motif_node_indices: list[int]  # indices of motif nodes after permutation
    motif_edges: list[tuple[int, int]]  # directed edges (src, dst) in permuted space
    class_label: str  # human-readable class name
    coupling_values: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ar1_nuisance(
    T: int, n: int, rho: float, noise_std: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate n independent AR(1) nuisance channels, shape (T, n)."""
    if n <= 0:
        return np.empty((T, 0))
    X = np.zeros((T, n))
    X[0] = rng.normal(0, noise_std, size=n)
    for t in range(1, T):
        X[t] = rho * X[t - 1] + rng.normal(0, noise_std, size=n)
    return X


def _permute_and_merge(
    motif_data: np.ndarray,
    nuisance_data: np.ndarray,
    motif_edges: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[int], list[tuple[int, int]]]:
    """
    Merge motif channels with nuisance channels, then randomly permute
    all channel indices. Returns (data, permuted_motif_indices, permuted_edges).
    """
    n_motif = motif_data.shape[1]
    n_nuisance = nuisance_data.shape[1]
    M = n_motif + n_nuisance
    combined = np.column_stack([motif_data, nuisance_data]) if n_nuisance > 0 else motif_data

    perm = rng.permutation(M)
    inv_perm = np.argsort(perm)  # not needed; we map old→new via perm

    data_permuted = combined[:, perm]

    # Build old-index → new-index map
    old_to_new = np.empty(M, dtype=int)
    for new_idx, old_idx in enumerate(perm):
        old_to_new[old_idx] = new_idx

    motif_indices = [int(old_to_new[i]) for i in range(n_motif)]
    edges_permuted = [(int(old_to_new[s]), int(old_to_new[d])) for s, d in motif_edges]

    return data_permuted, motif_indices, edges_permuted


# ---------------------------------------------------------------------------
# Generator A: directed VAR motifs (chain / fork / collider)
# ---------------------------------------------------------------------------

_CHAT_A_CLASSES = {
    0: "chain",     # 0→1→2
    1: "fork",      # 0→1, 0→2
    2: "collider",  # 1→0, 2→0
}


def _chat_a_motif_edges(motif_class: int) -> list[tuple[int, int]]:
    """Return directed edges for the 3-node motif in local (pre-permutation) space."""
    if motif_class == 0:  # chain: 0→1→2
        return [(0, 1), (1, 2)]
    if motif_class == 1:  # fork: 0→1, 0→2
        return [(0, 1), (0, 2)]
    if motif_class == 2:  # collider: 1→0, 2→0
        return [(1, 0), (2, 0)]
    raise ValueError(f"Unknown motif_class {motif_class}; expected 0, 1, or 2")


def generate_var_chat_a(
    M: int,
    T: int,
    motif_class: int = 0,
    alpha_lo: float = 0.25,
    alpha_hi: float = 0.8,
    rho_nuisance: float = 0.5,
    noise_std: float = 0.2,
    transients: int = 200,
    rng=None,
    zscore: bool = True,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, ChatMotifInternals]:
    """
    Generator A: embed a directed 3-node VAR motif into M nodes.

    Classes:
        0 = chain   (0→1→2)
        1 = fork    (0→1, 0→2)
        2 = collider (1→0, 2→0)

    Coupling strength α is drawn from Uniform(alpha_lo, alpha_hi) per sample.
    Couplings are positive. Remaining M-3 nodes are independent AR(1).

    Returns shape (T, M), or (data, ChatMotifInternals) if return_internals=True.
    """
    if M < 3:
        raise ValueError(f"M must be >= 3 for a 3-node motif, got {M}")
    rng = _resolve_rng(None, rng)

    edges = _chat_a_motif_edges(motif_class)
    alpha = float(rng.uniform(alpha_lo, alpha_hi))

    # Build 3x3 VAR(1) coefficient matrix
    A = np.zeros((3, 3))
    for src, dst in edges:
        A[dst, src] = alpha  # A[i,j] means j→i

    # Ensure stability: scale so spectral radius < 1
    sr = np.max(np.abs(np.linalg.eigvals(A)))
    if sr >= 0.99:
        A *= 0.95 / sr

    # Simulate motif
    steps = transients + T
    X_motif = np.zeros((steps, 3))
    for t in range(1, steps):
        X_motif[t] = A @ X_motif[t - 1] + rng.normal(0, noise_std, size=3)
    X_motif = X_motif[transients:]

    # Nuisance nodes
    X_nuisance = _ar1_nuisance(T, M - 3, rho_nuisance, noise_std, rng)

    # Permute
    data, motif_indices, edges_permuted = _permute_and_merge(
        X_motif, X_nuisance, edges, rng
    )
    data = _maybe_zscore(data, zscore=zscore)

    if return_internals:
        internals = ChatMotifInternals(
            motif_node_indices=motif_indices,
            motif_edges=edges_permuted,
            class_label=_CHAT_A_CLASSES[motif_class],
            coupling_values={"alpha": alpha},
        )
        return data, internals
    return data


# ---------------------------------------------------------------------------
# Generator B: common-driver confounder
# ---------------------------------------------------------------------------

_CHAT_B_CLASSES = {
    0: "no_direct",   # no direct 2→3 edge
    1: "with_direct",  # adds 2→3 edge
}


def generate_var_chat_b(
    M: int,
    T: int,
    motif_class: int = 0,
    a_lo: float = 0.25,
    a_hi: float = 0.8,
    b_lo: float = 0.25,
    b_hi: float = 0.8,
    c_lo: float = 0.25,
    c_hi: float = 0.8,
    rho: float = 0.5,
    noise_std: float = 0.2,
    transients: int = 200,
    rng=None,
    zscore: bool = True,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, ChatMotifInternals]:
    """
    Generator B: common-driver confounder with 3 motif nodes.

    Node 0 is the driver:
        x0(t) = rho * x0(t-1) + noise
        x1(t) = rho * x1(t-1) + a * x0(t-1) + noise
        x2(t) = rho * x2(t-1) + b * x0(t-1) + noise

    Classes:
        0 = no direct edge between nodes 1 and 2
        1 = adds x2(t) += c * x1(t-1)  (direct 1→2 link)

    Coupling strengths a, b, c drawn from Uniform ranges per sample.
    """
    if M < 3:
        raise ValueError(f"M must be >= 3 for a 3-node motif, got {M}")
    rng = _resolve_rng(None, rng)

    a = float(rng.uniform(a_lo, a_hi))
    b = float(rng.uniform(b_lo, b_hi))
    c = float(rng.uniform(c_lo, c_hi)) if motif_class == 1 else 0.0

    # Motif edges: 0→1, 0→2 always; 1→2 only in class 1
    edges: list[tuple[int, int]] = [(0, 1), (0, 2)]
    if motif_class == 1:
        edges.append((1, 2))

    # Build 3x3 VAR(1) coefficient matrix
    A = np.diag([rho, rho, rho])
    A[1, 0] = a   # 0→1
    A[2, 0] = b   # 0→2
    if motif_class == 1:
        A[2, 1] = c  # 1→2

    # Ensure stability
    sr = np.max(np.abs(np.linalg.eigvals(A)))
    if sr >= 0.99:
        A *= 0.95 / sr

    # Simulate motif
    steps = transients + T
    X_motif = np.zeros((steps, 3))
    for t in range(1, steps):
        X_motif[t] = A @ X_motif[t - 1] + rng.normal(0, noise_std, size=3)
    X_motif = X_motif[transients:]

    # Nuisance nodes
    X_nuisance = _ar1_nuisance(T, M - 3, rho, noise_std, rng)

    # Permute
    data, motif_indices, edges_permuted = _permute_and_merge(
        X_motif, X_nuisance, edges, rng
    )
    data = _maybe_zscore(data, zscore=zscore)

    if return_internals:
        coupling = {"a": a, "b": b, "rho": rho}
        if motif_class == 1:
            coupling["c"] = c
        internals = ChatMotifInternals(
            motif_node_indices=motif_indices,
            motif_edges=edges_permuted,
            class_label=_CHAT_B_CLASSES[motif_class],
            coupling_values=coupling,
        )
        return data, internals
    return data


# ---------------------------------------------------------------------------
# Generator C: nonlinear coupling
# ---------------------------------------------------------------------------

_CHAT_C_CLASSES = {
    0: "linear",   # g(u) = c*u
    1: "tanh",     # g(u) = tanh(c*u)
}


def generate_var_chat_c(
    M: int,
    T: int,
    motif_class: int = 0,
    c_lo: float = 0.5,
    c_hi: float = 1.5,
    rho: float = 0.5,
    noise_std: float = 0.2,
    transients: int = 200,
    rng=None,
    zscore: bool = True,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, ChatMotifInternals]:
    """
    Generator C: 2-node motif with linear vs nonlinear coupling.

    Node 0 is the driver:
        x0(t) = rho * x0(t-1) + noise
        x1(t) = rho * x1(t-1) + g(x0(t-1)) + noise

    Classes:
        0 = g(u) = c*u         (linear)
        1 = g(u) = tanh(c*u)   (nonlinear)

    Coupling c drawn from Uniform(c_lo, c_hi) per sample.
    """
    if M < 2:
        raise ValueError(f"M must be >= 2 for a 2-node motif, got {M}")
    rng = _resolve_rng(None, rng)

    c = float(rng.uniform(c_lo, c_hi))
    edges: list[tuple[int, int]] = [(0, 1)]

    if motif_class == 0:
        g = lambda u: c * u
    elif motif_class == 1:
        g = lambda u: np.tanh(c * u)
    else:
        raise ValueError(f"Unknown motif_class {motif_class}; expected 0 or 1")

    # Simulate 2-node motif
    steps = transients + T
    X_motif = np.zeros((steps, 2))
    for t in range(1, steps):
        X_motif[t, 0] = rho * X_motif[t - 1, 0] + rng.normal(0, noise_std)
        X_motif[t, 1] = rho * X_motif[t - 1, 1] + g(X_motif[t - 1, 0]) + rng.normal(0, noise_std)

    X_motif = X_motif[transients:]

    # Nuisance nodes
    X_nuisance = _ar1_nuisance(T, M - 2, rho, noise_std, rng)

    # Permute
    data, motif_indices, edges_permuted = _permute_and_merge(
        X_motif, X_nuisance, edges, rng
    )
    data = _maybe_zscore(data, zscore=zscore)

    if return_internals:
        internals = ChatMotifInternals(
            motif_node_indices=motif_indices,
            motif_edges=edges_permuted,
            class_label=_CHAT_C_CLASSES[motif_class],
            coupling_values={"c": c, "rho": rho},
        )
        return data, internals
    return data


# ---------------------------------------------------------------------------
# Generator D: lag discrimination
# ---------------------------------------------------------------------------

_CHAT_D_CLASSES = {
    0: "lag1",
    1: "lag3",
}


def generate_var_chat_d(
    M: int,
    T: int,
    motif_class: int = 0,
    alpha_lo: float = 0.25,
    alpha_hi: float = 0.8,
    rho: float = 0.5,
    noise_std: float = 0.2,
    transients: int = 200,
    rng=None,
    zscore: bool = True,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, ChatMotifInternals]:
    """
    Generator D: 2-node motif with coupling at different lags.

    Node 0 is the driver:
        x0(t) = rho * x0(t-1) + noise
        x1(t) = rho * x1(t-1) + alpha * x0(t - lag) + noise

    Classes:
        0 = lag 1
        1 = lag 3

    Same topology and coupling strength — only the lag differs.
    Tests whether temporal SPIs (lagged correlation, TLMI, TE) can
    discriminate lag structure.
    """
    if M < 2:
        raise ValueError(f"M must be >= 2 for a 2-node motif, got {M}")
    rng = _resolve_rng(None, rng)

    alpha = float(rng.uniform(alpha_lo, alpha_hi))
    lag = 1 if motif_class == 0 else 3
    edges: list[tuple[int, int]] = [(0, 1)]

    steps = transients + T
    X_motif = np.zeros((steps, 2))
    for t in range(lag, steps):
        X_motif[t, 0] = rho * X_motif[t - 1, 0] + rng.normal(0, noise_std)
        X_motif[t, 1] = rho * X_motif[t - 1, 1] + alpha * X_motif[t - lag, 0] + rng.normal(0, noise_std)
    X_motif = X_motif[transients:]

    X_nuisance = _ar1_nuisance(T, M - 2, rho, noise_std, rng)

    data, motif_indices, edges_permuted = _permute_and_merge(
        X_motif, X_nuisance, edges, rng
    )
    data = _maybe_zscore(data, zscore=zscore)

    if return_internals:
        internals = ChatMotifInternals(
            motif_node_indices=motif_indices,
            motif_edges=edges_permuted,
            class_label=_CHAT_D_CLASSES[motif_class],
            coupling_values={"alpha": alpha, "lag": lag, "rho": rho},
        )
        return data, internals
    return data
