from __future__ import annotations

from typing import Callable

import numpy as np

from ._common import _maybe_zscore, _resolve_rng


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic sigmoid."""
    x = np.asarray(x)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _E_softplus_gaussian(sigma2: float | np.ndarray, *, n_gh: int = 40) -> np.ndarray:
    """
    Approx E[softplus(Z)] for Z ~ N(0, sigma2) using Gauss-Hermite quadrature.
    Vectorized over sigma2.
    """
    s2 = np.asarray(sigma2, dtype=float)
    x, w = np.polynomial.hermite.hermgauss(n_gh)
    scale = np.sqrt(2.0 * s2)[..., None]
    vals = _softplus(scale * x[None, :])
    return (vals @ w) / np.sqrt(np.pi)


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
        beta = float(p.get("beta", 0.0))
        return lambda z, _a=alpha, _b=beta: _a * z + _b
    if name == "tanh":
        return np.tanh
    if name == "sin":
        return np.sin
    if name in {"sigmoid", "logistic"}:
        if "beta" not in p:
            raise ValueError("g='sigmoid' requires g_params with key 'beta' (slope).")
        beta = float(p["beta"])
        bias = float(p.get("bias", 0.0))
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
    if name == "exp":
        return lambda z: np.exp(np.clip(z, -50.0, 50.0))
    if name in {"exp_centered", "exp_shifted"}:
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
      - Only allowed when c == 0 (purely linear system).

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
                A[idx[1:], idx[:-1]] = b_arr[1:]
        else:
            raise ValueError("topology must be 'ring' or 'chain'")

        ev = np.linalg.eigvals(A)
        sr = float(np.max(np.abs(ev))) if ev.size else 0.0
        if sr >= float(target_rho) and sr > 0.0:
            scale = float(target_rho) / sr
            a_arr *= scale
            b_arr *= scale

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

    rng = _resolve_rng(None, rng)

    a_arr = np.broadcast_to(np.asarray(a, dtype=float), (M,)).copy()
    b_arr = np.broadcast_to(np.asarray(b, dtype=float), (M,)).copy()
    noise_std_arr = np.broadcast_to(np.asarray(noise_std, dtype=float), (M,)).copy()

    sigma2_auto = None
    g_name = str(g).lower().strip() if not callable(g) else ""
    if sigma2 is None and g_name.endswith(("centered", "shifted")):
        denom = 1.0 - a_arr**2
        if np.any(denom <= 0):
            raise ValueError("Cannot auto-compute sigma2 when |a|>=1.")
        sigma2_auto = (noise_std_arr**2) / denom

    g_fn = _resolve_g(g, sigma2=sigma2 if sigma2 is not None else sigma2_auto)

    if target_rho is not None:
        maxabs = float(np.max(np.abs(a_arr))) if a_arr.size else 0.0
        if maxabs > float(target_rho) and maxabs > 0.0:
            a_arr *= float(target_rho) / maxabs

    total_T = T + transients
    z = np.zeros((total_T, M), dtype=float)

    if z0 is None:
        z[0] = rng.normal(scale=float(init_std), size=(M,))
    else:
        z[0] = np.broadcast_to(np.asarray(z0, dtype=float), (M,))

    eps = rng.normal(size=(max(total_T - 1, 0), M)) * noise_std_arr

    for t in range(1, total_T):
        z[t] = a_arr * z[t - 1] + eps[t - 1]

    topology_key = str(topology).lower().strip()
    if topology_key == "ring":
        prev = np.roll(z, 1, axis=1)
    elif topology_key == "chain":
        prev = np.empty_like(z)
        prev[:, 0] = float(boundary_value)
        prev[:, 1:] = z[:, :-1]
    else:
        raise ValueError("topology must be 'ring' or 'chain'")

    x = z + g_fn(prev) * b_arr

    x_out = x[transients:] if transients > 0 else x
    z_out = z[transients:] if transients > 0 else z

    x_out = _maybe_zscore(x_out, zscore=zscore)
    if return_latents:
        z_out = _maybe_zscore(z_out, zscore=zscore)
        return x_out, z_out

    return x_out


def generate_case_iii(
    M: int,
    T: int,
    *,
    mode: str = "lagged_drive_pairs",
    a_driver: float | np.ndarray = 0.0,
    a_resp: float | np.ndarray = 0.6,
    coupling: float | np.ndarray = 0.8,
    noise_std_driver: float | np.ndarray = 1.0,
    noise_std_resp: float | np.ndarray = 1.0,
    a_base: float | np.ndarray = 0.6,
    noise_std_base: float | np.ndarray = 1.0,
    latent_std: float = 1.0,
    alpha: float | np.ndarray = 1.0,
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

    mode="instantaneous_common_cause":
        Independent AR(1) bases + instantaneous i.i.d. common cause.

    Returns array shape (T, M).
    """
    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if transients < 0:
        raise ValueError(f"transients must be >= 0, got {transients}")

    rng = _resolve_rng(None, rng)

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

        drivers = (np.arange(M) % 2 == 0)
        responders = ~drivers

        eps = rng.normal(size=(max(total_T - 1, 0), M))
        for t in range(1, total_T):
            prev = x[t - 1]

            nxt = np.empty_like(prev)
            nxt[drivers] = aD[drivers] * prev[drivers] + eps[t - 1, drivers] * sD[drivers]

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

        eps = rng.normal(size=(max(total_T - 1, 0), M))
        for t in range(1, total_T):
            x[t] = a * x[t - 1] + eps[t - 1] * s

        Z = rng.normal(scale=float(latent_std), size=(total_T,))
        x = x + Z[:, None] * alpha_arr[None, :]

    else:
        raise ValueError("mode must be one of: 'lagged_drive_pairs', 'instantaneous_common_cause'")

    out = x[transients:] if transients > 0 else x
    return _maybe_zscore(out, zscore=zscore)


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
        Master signal:  z_{t+1} = a * z_t + eps_t,  where eps_t ~ N(0, noise_std^2)
        Observations:   X_t^{(i)} = g(z_t; beta) + alpha * eta_t^{(i)},  where eta ~ N(0, 1)
    """
    if M <= 0:
        raise ValueError("M must be positive")
    if T <= 0:
        raise ValueError("T must be positive")

    rng = _resolve_rng(seed, rng)

    total_T = T + transients
    z = np.zeros(total_T, dtype=float)
    eps = rng.normal(0, noise_std, size=total_T)
    for t in range(1, total_T):
        z[t] = a * z[t - 1] + eps[t]

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

    z_use = z[transients:]
    filtered = g_fn(z_use)

    X = np.tile(filtered[:, None], (1, M))
    if alpha > 0:
        eta = rng.normal(0, 1, size=(T, M))
        X = X + alpha * eta

    return _maybe_zscore(X, zscore=zscore)
