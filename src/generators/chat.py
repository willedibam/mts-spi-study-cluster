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
    lag: int = 1,
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

    `lag` sets the delay of the coupling: X[t] = A X[t-lag] + e, so the CORRECT
    autoregressive model order for this process is exactly `lag`. Default 1
    reproduces every existing dataset byte-for-byte.

    This exists to test a prediction the probe makes. On R0 (lag=1) the learned
    weights prefer `sgc_parametric` at order-1 over the SAME estimator at
    order-20 by ~8.6x, in 10/10 lambda runs -- a statistical-efficiency property
    (order-1 is correctly specified) that the generator does not encode. If that
    reading is right, raising `lag` must move the preference toward the higher
    orders. A preference that stays on order-1 regardless of `lag` falsifies it.

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
    for t in range(lag, steps):
        X_motif[t] = A @ X_motif[t - lag] + rng.normal(0, noise_std, size=3)
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
    for t in range(lag, steps):
        X_motif[t] = A @ X_motif[t - lag] + rng.normal(0, noise_std, size=3)
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


def _match_ar1(X: np.ndarray, rho_target: float) -> np.ndarray:
    """Give every column the same lag-1 autocorrelation, preserving coupling.

    Whitens each channel with its own estimated AR(1) coefficient, then
    re-colours with `rho_target`. Per-channel filtering leaves the cross-channel
    dependence structure in place while removing the marginal signature that
    would otherwise let node-level features identify the motif.
    """
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        x = X[:, j] - X[:, j].mean()
        c0 = float(x @ x)
        a = float(x[:-1] @ x[1:]) / c0 if c0 > 1e-12 else 0.0
        a = float(np.clip(a, -0.99, 0.99))
        w = np.empty_like(x)
        w[0] = x[0]
        w[1:] = x[1:] - a * x[:-1]            # whiten
        y = np.empty_like(w)
        y[0] = w[0]
        for t in range(1, len(w)):            # re-colour to the shared target
            y[t] = rho_target * y[t - 1] + w[t]
        out[:, j] = y
    return out


# ---------------------------------------------------------------------------
# Generator E: nonlinear-directed motifs (R1 of the multi-regime study)
# ---------------------------------------------------------------------------

_CHAT_E_CLASSES = {
    0: "chain",     # 0→1→2
    1: "fork",      # 0→1, 0→2
    2: "collider",  # 1→0, 2→0
}


def generate_var_nonlinear_a(
    M: int,
    T: int,
    motif_class: int = 0,
    alpha_lo: float = 0.3,
    alpha_hi: float = 0.7,
    rho_nuisance: float = 0.8,
    noise_std: float = 0.1,
    gain: float = 2.0,
    coupling: str = "square",
    match_marginals: bool = False,
    transients: int = 200,
    rng=None,
    zscore: bool = True,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, ChatMotifInternals]:
    """
    Generator E (R1): the var_chat_a motifs with a NONLINEAR directed coupling.

    Identical topology, embedding and nuisance structure to generate_var_chat_a
    -- only the coupling function changes:

        child(t) = rho*child(t-1) + alpha*tanh(gain * parent(t-1)) + noise

    Purpose. On the linear VAR (var_chat_a), a Granger-family statistic MUST
    win: linear autoregression is exactly what GC detects, so "w recovers GC"
    is confirmation, not discovery. Here the directed dependence is monotone
    but nonlinear and saturating, so linear GC is mis-specified while
    transfer entropy / directed information are not.

    Pre-registered prediction: within the causal family, TE / directed-info
    (kraskov estimators) should outrank linear GC. If linear GC still wins,
    the method cannot see nonlinear direction -- a real, falsifiable failure.

    coupling:
      "square" (default) -- child gets alpha * (parent(t-1))**2. NON-monotone,
          so Pearson/linear-GC on parent->child is ~0 by symmetry while the
          dependence is total. This is the sharp test.
      "tanh" -- child gets alpha * tanh(gain * parent(t-1)). Monotone and
          saturating; measured on this generator it leaves linear GC almost
          as good as a nonlinear measure (corr gap < 0.02 even at gain 40),
          so it is a WEAK test of nonlinear-direction sensitivity. Kept for
          completeness; do not use it as the headline R1 regime.

    `gain` applies to the tanh variant only.
    """
    if coupling not in ("square", "tanh"):
        raise ValueError(f"coupling must be 'square' or 'tanh', got {coupling!r}")

    # match_marginals: equalise every channel's lag-1 autocorrelation to a
    # common target, so per-channel dynamics cannot identify the motif.
    #
    # Needed because a squared drive changes a node's effective AR structure by
    # an amount that depends on its in-degree, and the motifs differ there:
    # chain and fork both have degrees [0,1,1] but the collider is [2,0,0] --
    # one node absorbs two squared drives. Measured on the unfixed generator
    # (M=10, T=1000, 60 draws), the top-3 lag-1 AC profile is a fingerprint:
    #   chain    [0.974, 0.945, 0.821]
    #   fork     [0.947, 0.946, 0.821]
    #   collider [0.948, 0.824, 0.812]
    # and a logistic regression on node features alone separates chain from
    # collider at 0.83 (chance 0.5), driven almost entirely by lag-1 AC (0.85).
    # That makes the classification solvable WITHOUT any pairwise coupling,
    # which would invalidate any claim about the recovered coupling signature.
    #
    # The correction whitens each channel with its own estimated AR(1)
    # coefficient and re-colours with the shared target, which removes the
    # motif-dependent marginal while leaving cross-channel structure intact.
    if M < 3:
        raise ValueError(f"M must be >= 3 for a 3-node motif, got {M}")
    rng = _resolve_rng(None, rng)

    edges = _chat_a_motif_edges(motif_class)
    alpha = float(rng.uniform(alpha_lo, alpha_hi))

    # Parents of each motif node, in local (pre-permutation) space.
    parents: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for src, dst in edges:
        parents[dst].append(src)

    # Stationary variance of the AR(1) baseline, used to normalise the
    # quadratic drive to unit scale.
    _sigma2 = noise_std ** 2 / max(1.0 - rho_nuisance ** 2, 1e-6)

    steps = transients + T
    X_motif = np.zeros((steps, 3))
    for t in range(1, steps):
        for node in range(3):
            if coupling == "square":
                # Non-monotone: linear GC cannot see it, TE can. The square is
                # centred and scaled by the stationary AR(1) variance so the
                # drive is O(1) rather than O(sigma^2) -- otherwise the signal
                # is swamped by the noise term and BOTH linear and nonlinear
                # measures see nothing (measured: MI 0.03 nats unnormalised).
                drive = sum(
                    (X_motif[t - 1, p] ** 2 - _sigma2) / _sigma2
                    for p in parents[node]
                )
            else:
                drive = sum(np.tanh(gain * X_motif[t - 1, p]) for p in parents[node])
            X_motif[t, node] = (
                rho_nuisance * X_motif[t - 1, node]
                + alpha * drive
                + rng.normal(0, noise_std)
            )
    X_motif = X_motif[transients:]

    if match_marginals:
        X_motif = _match_ar1(X_motif, rho_nuisance)

    X_nuisance = _ar1_nuisance(T, M - 3, rho_nuisance, noise_std, rng)

    data, motif_indices, edges_permuted = _permute_and_merge(
        X_motif, X_nuisance, edges, rng
    )
    data = _maybe_zscore(data, zscore=zscore)

    if return_internals:
        internals = ChatMotifInternals(
            motif_node_indices=motif_indices,
            motif_edges=edges_permuted,
            class_label=_CHAT_E_CLASSES[motif_class],
            coupling_values={"alpha": alpha, "gain": gain,
                             "rho": rho_nuisance, "coupling": coupling},
        )
        return data, internals
    return data


# ---------------------------------------------------------------------------
# Generator F: linear VAR with a NON-MONOTONE static observation (R1b)
# ---------------------------------------------------------------------------

def generate_var_obs_nonlinear_a(
    M: int,
    T: int,
    motif_class: int = 0,
    observation: str = "square",
    alpha_lo: float = 0.2,
    alpha_hi: float = 0.8,
    rho_nuisance: float = 0.8,
    noise_std: float = 0.1,
    transients: int = 200,
    rng=None,
    zscore: bool = True,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, ChatMotifInternals]:
    """R1b: var_chat_a dynamics, observed through a non-monotone transform.

    Latent dynamics are the LINEAR VAR of generate_var_chat_a; only the
    observation is nonlinear, and the same transform is applied to every
    channel of every motif.

    Why this rather than a nonlinear coupling. generate_var_nonlinear_a injects
    the nonlinearity into the update, which changes each node's effective AR
    structure by an amount that depends on its in-degree -- and the motifs
    differ there (chain and fork are [0,1,1] but the collider is [2,0,0]). The
    marginals then identify the motif on their own: measured with node features
    alone, chain vs collider is separable at 0.91 (chance 0.5), so the task can
    be solved with NO pairwise information and any claim about a recovered
    coupling signature is void.

    Applying the nonlinearity at the observation stage cannot leak, because it
    is motif-independent by construction. Measured the same way:
        R0 linear VAR (clean reference)  0.54 / 0.54
        R1  nonlinear coupling           0.91 / 0.58   <- confounded
        R1b linear VAR + x^2 observation 0.47 / 0.49   <- at chance

    Trade-off, stated honestly: linear statistics are MIS-SPECIFIED here rather
    than blind. For jointly Gaussian latents, corr(x_i^2, x_j^2) = 2*corr^2, so
    linear measures still detect coupling, only attenuated. The prediction is
    therefore that nonlinear/information-theoretic measures gain RELATIVE
    weight, not that linear ones collapse to zero -- a weaker claim than the
    confounded design appeared to support, but one that survives its controls.

    observation: "square" (x^2, non-monotone) or "abs" (|x|).
    """
    if observation not in ("square", "abs"):
        raise ValueError(f"observation must be 'square' or 'abs', got {observation!r}")

    out = generate_var_chat_a(
        M=M, T=T, motif_class=motif_class,
        alpha_lo=alpha_lo, alpha_hi=alpha_hi,
        rho_nuisance=rho_nuisance, noise_std=noise_std,
        transients=transients, rng=rng, zscore=False,
        return_internals=return_internals,
    )
    data, internals = (out if return_internals else (out, None))
    data = np.square(data) if observation == "square" else np.abs(data)
    data = _maybe_zscore(data, zscore=zscore)

    if return_internals:
        internals.coupling_values["observation"] = observation
        return data, internals
    return data
