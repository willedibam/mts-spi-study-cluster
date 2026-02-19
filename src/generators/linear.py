from __future__ import annotations

import numpy as np

from ._common import _maybe_zscore, _resolve_rng


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
    topology: str = "ring-symmetric",
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
    right = np.roll(I, 1, axis=1)
    left = np.roll(I, -1, axis=1)
    if topology == "ring-symmetric":
        neighbors = left + right
    elif topology == "ring-unidirectional":
        neighbors = left
    elif topology == "all-to-all":
        neighbors = np.ones((M, M)) - I
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
    """Standard Exponential distribution (rate parameter gamma = 1)."""
    rng = _resolve_rng(None, rng)
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
    Discretised with Euler-Maruyama in log space:
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


def _warp_path(T: int, regime: str, *, step: int, p: float, p_step: float, rng) -> np.ndarray:
    """Random monotone warping path from (0,0) to (T-1,T-1)."""
    x, y = 0, 0
    path = [(x, y)]
    if regime in ("walk", "geometric"):
        while x < T - 1 or y < T - 1:
            s = step if regime == "walk" else int(np.clip(rng.geometric(p), 1, T))
            if x >= T - 1:
                y = min(y + s, T - 1)
            elif y >= T - 1:
                x = min(x + s, T - 1)
            elif rng.integers(2) == 0:
                x = min(x + s, T - 1)
            else:
                y = min(y + s, T - 1)
            path.append((x, y))
    elif regime == "rebound":
        while x < T - 1:
            s = int(min(rng.geometric(p), T - 1 - x))
            if rng.uniform() < p_step:
                if rng.integers(2) == 0:  # R then U
                    x += s; path.append((x, y))
                    y += s; path.append((x, y))
                else:                      # U then R
                    y += s; path.append((x, y))
                    x += s; path.append((x, y))
            else:
                x += 1; y += 1
                path.append((x, y))
    else:
        raise ValueError(f"warping_regime must be 'walk', 'geometric', or 'rebound', got {regime!r}")
    return np.asarray(path, dtype=int)


def _path_to_mapping(path: np.ndarray, T: int) -> np.ndarray:
    """Convert a warping path to a time-index mapping array of length T."""
    n = np.zeros(T, dtype=int)
    s = np.zeros(T, dtype=float)
    for x, y in path:
        n[y] += 1
        s[y] += x
    m = np.zeros(T, dtype=int)
    for y in range(T):
        m[y] = int(round(s[y] / n[y])) if n[y] > 0 else 0
    m = np.clip(m, 0, T - 1)
    m = np.maximum.accumulate(m)
    return m


def generate_warping_mts(
    M: int,
    T: int,
    regime: str = "fixed",
    n_noisy: int | None = None,
    n_warped: int | None = None,
    warping_regime: str = "rebound",
    warp_p: float = 0.5,
    warp_p_step: float = 0.5,
    warp_step: int = 1,
    noise_std: float = 0.05,
    ar1_a: float = 0.8,
    ar1_noise_std: float = 1.0,
    zscore: bool = False,
    _regime_seed: int | None = None,
    rng=None,
) -> np.ndarray:
    """
    M-channel MTS built from a shared AR(1) mother signal. Each channel is either
    a noisy copy of the mother or a time-warped copy with added noise.

    Channel assignment
    ------------------
    "fixed"  : first n_noisy channels are noisy copies, next n_warped are warped.
               Defaults: n_noisy = M // 2, n_warped = M - M // 2.
    "random" : each channel independently and uniformly assigned noisy or warped (50/50).
               Pass _regime_seed for reproducible assignment independent of data noise seed.

    Warping regimes (warping_regime)
    ---------------------------------
    "walk"      : fixed step size warp_step, uniform random direction.
    "geometric" : step ~ Geometric(warp_p), uniform random direction.
    "rebound"   : step ~ Geometric(warp_p), L-shaped excursions with prob warp_p_step,
                  else diagonal step. Always returns to identity line after each cycle.

    Parameters
    ----------
    M              : number of channels
    T              : time steps
    regime         : "fixed" | "random"  — channel assignment
    n_noisy        : noisy channels   (fixed only; default M // 2)
    n_warped       : warped channels  (fixed only; default M - M // 2)
    warping_regime : warp path type   "walk" | "geometric" | "rebound"
    warp_p         : Geometric parameter       (geometric, rebound)
    warp_p_step    : L-excursion probability   (rebound only)
    warp_step      : fixed step size           (walk only)
    noise_std      : noise std added to every channel
    ar1_a          : AR(1) autoregressive coefficient for mother signal
    ar1_noise_std  : noise std of the mother signal
    zscore         : z-score output
    _regime_seed   : seed for channel assignment RNG  (random regime)
    rng            : RNG instance or int seed

    Returns shape (T, M).
    Slug convention: M<M>_T<T>_I<I>_m<n_noisy>_w<n_warped>
    """
    rng = _resolve_rng(None, rng)

    # Mother signal: AR(1)
    mother = np.zeros(T)
    for t in range(1, T):
        mother[t] = ar1_a * mother[t - 1] + rng.normal(0, ar1_noise_std)

    # Channel assignment
    if regime == "fixed":
        nm = int(n_noisy)  if n_noisy  is not None else M // 2
        nw = int(n_warped) if n_warped is not None else M - nm
        if nm + nw != M:
            raise ValueError(f"n_noisy ({nm}) + n_warped ({nw}) != M ({M}).")
        is_warped = [False] * nm + [True] * nw
    elif regime == "random":
        regime_rng = np.random.default_rng(_regime_seed) if _regime_seed is not None else rng
        is_warped = [bool(regime_rng.integers(2)) for _ in range(M)]
    else:
        raise ValueError(f"regime must be 'fixed' or 'random', got {regime!r}")

    data = np.empty((T, M))
    for i in range(M):
        if is_warped[i]:
            path = _warp_path(T, warping_regime, step=warp_step, p=warp_p, p_step=warp_p_step, rng=rng)
            mapping = _path_to_mapping(path, T)
            data[:, i] = mother[mapping] + rng.normal(0, noise_std, T)
        else:
            data[:, i] = mother + rng.normal(0, noise_std, T)

    return _maybe_zscore(data, zscore=zscore)


_SIN_INTERVALS = {
    "linear":        (-np.pi / 16, np.pi / 16),
    "monotonic":     (-np.pi / 2,  np.pi / 2),
    "non-monotonic": (-np.pi,       np.pi),
}


def _sin_channels(data: np.ndarray, channel_regimes: list, T: int, noise_std: float, rng) -> None:
    """Fill data[:, i] in-place for each channel given its regime."""
    for i, reg in enumerate(channel_regimes):
        lo, hi = _SIN_INTERVALS[reg]
        t = np.linspace(lo, hi, T)
        data[:, i] = np.sin(t) + rng.normal(0, noise_std, T)


def generate_sin_mts(
    M: int,
    T: int,
    regime: str = "fixed",
    n_linear: int | None = None,
    n_monotonic: int | None = None,
    noise_std: float = 0.05,
    zscore: bool = False,
    _regime_seed: int | None = None,
    rng=None,
) -> np.ndarray:
    """
    M sinusoidal channels, each sampled over a regime-specific interval of sin(t).

    Regimes (interval of t for sin(t)):
      linear:        t in [-pi/16, pi/16]  — approximately linear
      monotonic:     t in [-pi/2,  pi/2]   — strictly increasing
      non-monotonic: t in [-pi,    pi]     — full cycle

    regime='fixed':  ordered assignment — first n_linear channels linear, next n_monotonic
                     monotonic, remaining non-monotonic. Defaults to M//3 each (3+3+3 for M=9).
    regime='random': each channel randomly assigned. When run via the experiment framework,
                     _regime_seed is injected from (M, T, instance, mts_class) so the slug is
                     reproducible without coupling to the dataset noise seed.

    Returns shape (T, M).
    """
    rng = _resolve_rng(None, rng)
    if regime == "fixed":
        nl = int(n_linear) if n_linear is not None else M // 3
        nm = int(n_monotonic) if n_monotonic is not None else M // 3
        nnm = M - nl - nm
        if nnm < 0:
            raise ValueError(f"n_linear ({nl}) + n_monotonic ({nm}) > M ({M}).")
        channel_regimes = ["linear"] * nl + ["monotonic"] * nm + ["non-monotonic"] * nnm
    elif regime == "random":
        keys = list(_SIN_INTERVALS.keys())
        regime_rng = np.random.default_rng(_regime_seed) if _regime_seed is not None else rng
        channel_regimes = [keys[i] for i in regime_rng.integers(0, 3, size=M)]
    else:
        raise ValueError(f"Unknown regime '{regime}'. Expected 'fixed' or 'random'.")
    data = np.empty((T, M))
    _sin_channels(data, channel_regimes, T, noise_std, rng)
    return _maybe_zscore(data, zscore=zscore)


