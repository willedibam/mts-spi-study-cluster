from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ._common import _maybe_zscore, _resolve_channel_noise_stds, _resolve_rng


def generate_varma(
    M: int,
    T: int,
    phi: float = 0.6,
    coupling: float = 0.4,
    ma_phi: float = 0.2,
    ma_coupling: float = 0.1,
    noise_std: float = 0.1,
    noise_std_variable: bool = False,
    noise_std_scale: float = np.e,
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
    noise_stds = _resolve_channel_noise_stds(
        noise_std, M, noise_std_variable=noise_std_variable,
        noise_std_scale=noise_std_scale, rng=rng,
    )
    eps = rng.normal(0.0, 1.0, size=(steps, M)) * noise_stds[None, :]
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
    noise_std_variable: bool = False,
    noise_std_scale: float = np.e,
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
        noise_std=noise_std, noise_std_variable=noise_std_variable,
        noise_std_scale=noise_std_scale, transients=transients,
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


@dataclass
class WarpingInternals:
    """Internals returned by generate_warping_mts when return_internals=True."""
    mother: np.ndarray                    # (T,) mother signal before channel noise
    is_warped: list                       # (M,) True if channel is a warped copy
    paths: list                           # (M,) warping path array or None per channel
    mappings: list                        # (M,) time-index mapping array or None per channel


def generate_warping_mts(
    M: int,
    T: int,
    warping_channel_regime: str = "fixed",
    n_noisy: int | None = None,
    n_warped: int | None = None,
    warping_path_regime: str = "rebound",
    warp_p: float = 0.5,
    warp_p_step: float = 0.5,
    warp_step: int = 1,
    noise_std: float = 0.05,
    noise_std_variable: bool = False,
    noise_std_scale: float = np.e,
    ar1_a: float = 0.8,
    ar1_noise_std: float = 1.0,
    zscore: bool = False,
    _regime_seed: int | None = None,
    rng=None,
    mother: np.ndarray | None = None,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, WarpingInternals]:
    """
    M-channel MTS built from a shared mother signal. Each channel is either
    a noisy copy of the mother or a time-warped copy with added noise.

    Channel assignment
    ------------------
    "fixed"  : first n_noisy channels are noisy copies, next n_warped are warped.
               Defaults: n_noisy = M // 2, n_warped = M - M // 2.
    "random" : each channel independently and uniformly assigned noisy or warped (50/50).
               Pass _regime_seed for reproducible assignment independent of data noise seed.

    Warping regimes (warping_path_regime)
    ---------------------------------
    "walk"      : fixed step size warp_step, uniform random direction.
    "geometric" : step ~ Geometric(warp_p), uniform random direction.
    "rebound"   : step ~ Geometric(warp_p), L-shaped excursions with prob warp_p_step,
                  else diagonal step. Always returns to identity line after each cycle.

    Parameters
    ----------
    M              : number of channels
    T              : time steps
    warping_channel_regime : "fixed" | "random"  — channel assignment
    n_noisy        : noisy channels   (fixed only; default M // 2)
    n_warped       : warped channels  (fixed only; default M - M // 2)
    warping_path_regime : warp path type   "walk" | "geometric" | "rebound"
    warp_p         : Geometric parameter controlling step size       (geometric, rebound)
    warp_p_step    : L-excursion probability   (rebound only)
    warp_step      : fixed step size           (walk only)
    noise_std      : noise std added to every channel
    ar1_a          : AR(1) autoregressive coefficient for mother signal (ignored if mother provided)
    ar1_noise_std  : noise std of the mother signal (ignored if mother provided)
    zscore         : z-score output
    _regime_seed   : seed for channel assignment RNG  (random regime)
    rng            : RNG instance or int seed
    mother         : optional 1D array of length T to use as the mother signal instead of AR(1)
    return_internals : if True, return (data, WarpingInternals) instead of just data

    Returns shape (T, M), or tuple of ((T, M), WarpingInternals) when return_internals=True.
    Slug convention: M<M>_T<T>_I<I>_m<n_noisy>_w<n_warped>
    """
    rng = _resolve_rng(None, rng)

    # Mother signal
    if mother is not None:
        mother = np.asarray(mother, dtype=float)
        if mother.ndim != 1 or len(mother) != T:
            raise ValueError(f"mother must be 1D of length T={T}, got shape {mother.shape}")
    else:
        mother = np.zeros(T)
        for t in range(1, T):
            mother[t] = ar1_a * mother[t - 1] + rng.normal(0, ar1_noise_std)

    # Channel assignment
    if warping_channel_regime == "fixed":
        nm = int(n_noisy)  if n_noisy  is not None else M // 2
        nw = int(n_warped) if n_warped is not None else M - nm
        if nm + nw != M:
            raise ValueError(f"n_noisy ({nm}) + n_warped ({nw}) != M ({M}).")
        is_warped = [False] * nm + [True] * nw
    elif warping_channel_regime == "random":
        regime_rng = np.random.default_rng(_regime_seed) if _regime_seed is not None else rng
        is_warped = [bool(regime_rng.integers(2)) for _ in range(M)]
    else:
        raise ValueError(f"warping_channel_regime must be 'fixed' or 'random', got {warping_channel_regime!r}")

    noise_stds = _resolve_channel_noise_stds(
        noise_std, M, noise_std_variable=noise_std_variable,
        noise_std_scale=noise_std_scale, rng=rng,
    )
    data = np.empty((T, M))
    paths: list = []
    mappings: list = []
    for i in range(M):
        if is_warped[i]:
            path = _warp_path(T, warping_path_regime, step=warp_step, p=warp_p, p_step=warp_p_step, rng=rng)
            mapping = _path_to_mapping(path, T)
            data[:, i] = mother[mapping] + rng.normal(0, noise_stds[i], T)
            paths.append(path)
            mappings.append(mapping)
        else:
            data[:, i] = mother + rng.normal(0, noise_stds[i], T)
            identity = np.arange(T)
            paths.append(np.column_stack([identity, identity]))
            mappings.append(identity)

    result = _maybe_zscore(data, zscore=zscore)
    if return_internals:
        return result, WarpingInternals(mother=mother, is_warped=is_warped, paths=paths, mappings=mappings)
    return result


@dataclass
class LaggedInternals:
    """Internals returned by generate_lagged_mts when return_internals=True."""
    mother: np.ndarray  # (T + max_lag,) full extended mother signal
    lags: np.ndarray    # (M,) integer lag tau_i per channel


def generate_lagged_mts(
    M: int,
    T: int,
    A: float = 1.0,
    linspace: bool = False,
    fixed_lag: bool = False,
    lag_dist: str = "geometric",
    noise_std: float = 0.05,
    noise_std_variable: bool = False,
    noise_std_scale: float = np.e,
    noise_dist: str = "geometric",
    ar1_a: float = 0.8,
    ar1_noise_std: float = 1.0,
    zscore: bool = False,
    rng=None,
    mother: np.ndarray | None = None,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, LaggedInternals]:
    """
    M-channel MTS built from a shared AR(1) mother signal, each channel
    shifted by an integer lag tau_i.

    Per-channel construction:
        X_t^{(i)} = mother[t - tau_i] + eta_t^{(i)},   eta ~ N(0, noise_std_i^2)

    Lag parameterisation (controlled by A >= 1); precedence: fixed_lag > linspace > random:
        fixed_lag=True:  tau_i = round(A-1) for every channel (all channels share
            the same deterministic lag; A=1 => tau=0, A=k => tau=k-1).
        linspace=True:   tau_i = round(linspace(0, A-1, M)[i])  (deterministic,
            evenly spaced from 0 to round(A-1)).
        default (both False):  controlled by lag_dist:
            lag_dist="geometric": tau_i ~ Geom(p=1/A) - 1  [original]
                A=1 => tau_i=0 always; A>1 => E[tau_i]=A-1, heavy right tail.
            lag_dist="uniform":   tau_i ~ DiscreteUniform(0, round(A-1))
                A=1 => tau_i=0 always; A>1 => E[tau_i]=(A-1)/2, bounded support.

    Noise std parameterisation (when noise_std_variable=True); controlled by noise_dist:
        noise_dist="geometric": noise_std_i = noise_std * kappa_i,
            kappa_i ~ Geom(1/noise_std_scale); integer multiples, E[std]=noise_std*scale. [original]
        noise_dist="uniform":   noise_std_i ~ Uniform(noise_std/noise_std_scale,
            noise_std*noise_std_scale); continuous, same effective range.

    The mother signal is extended by max_lag steps at the front so that every
    channel has exactly T valid time steps.

    Parameters
    ----------
    M             : number of channels
    T             : time steps in output
    A             : lag scale; must be >= 1. A=1 => no lag.
    linspace      : if True, lags evenly spaced 0..round(A-1) (ignored if fixed_lag)
    fixed_lag     : if True, all channels get the same lag tau = round(A-1)
    lag_dist      : "geometric" (original) | "uniform" — active only when linspace=False, fixed_lag=False
    noise_std     : per-channel i.i.d. measurement noise std (base value)
    noise_std_variable : if True, per-channel noise std sampled according to noise_dist
    noise_std_scale    : scale parameter for noise distribution
    noise_dist    : "geometric" (original) | "uniform" — active only when noise_std_variable=True
    ar1_a         : AR(1) autoregressive coefficient for mother signal
    ar1_noise_std : driving noise std of the mother signal
    zscore        : z-score output
    rng           : RNG instance or int seed
    mother        : optional 1D array of length >= T + max_lag to use as mother
    return_internals : if True, return (data, LaggedInternals)

    Returns shape (T, M), or tuple of ((T, M), LaggedInternals).
    """
    if A < 1.0:
        raise ValueError(f"A must be >= 1 (got {A}). A=1 means no lag.")
    if lag_dist not in ("geometric", "uniform"):
        raise ValueError(f"lag_dist must be 'geometric' or 'uniform', got {lag_dist!r}")
    if noise_dist not in ("geometric", "uniform"):
        raise ValueError(f"noise_dist must be 'geometric' or 'uniform', got {noise_dist!r}")
    rng = _resolve_rng(None, rng)

    # --- Assign per-channel lags ---
    if fixed_lag:
        tau = int(round(A - 1))
        lags = np.full(M, tau, dtype=int)
    elif linspace:
        max_lag = int(round(A - 1))
        lags = np.round(np.linspace(0, max_lag, M)).astype(int)
    elif lag_dist == "geometric":
        # Original: tau_i ~ Geom(1/A) - 1; E[tau]=A-1, heavy right tail, unbounded.
        p = 1.0 / A
        lags = rng.geometric(p, size=M) - 1
    else:
        # Uniform: tau_i ~ DiscreteUniform(0, round(A-1)); bounded support [0, A-1].
        max_lag_u = int(round(A - 1))
        lags = rng.integers(0, max_lag_u + 1, size=M)

    max_lag = int(lags.max())
    extended_T = T + max_lag

    # --- Mother signal ---
    if mother is not None:
        mother = np.asarray(mother, dtype=float)
        if mother.ndim != 1 or len(mother) < extended_T:
            raise ValueError(
                f"mother must be 1D of length >= T + max_lag = {extended_T}, "
                f"got shape {mother.shape}"
            )
    else:
        m = np.zeros(extended_T)
        for t in range(1, extended_T):
            m[t] = ar1_a * m[t - 1] + rng.normal(0, ar1_noise_std)
        mother = m

    # --- Per-channel noise stds ---
    if not noise_std_variable:
        noise_stds = np.full(M, float(noise_std))
    elif noise_dist == "geometric":
        # Original: noise_std_i = noise_std * kappa_i, kappa_i ~ Geom(1/scale);
        # discrete integer multiples of noise_std, E[std] = noise_std * scale.
        noise_stds = _resolve_channel_noise_stds(
            noise_std, M, noise_std_variable=True,
            noise_std_scale=noise_std_scale, rng=rng,
        )
    else:
        # Uniform: noise_std_i ~ Uniform(noise_std/scale, noise_std*scale);
        # continuous, symmetric range around noise_std on log scale.
        lo = float(noise_std) / float(noise_std_scale)
        hi = float(noise_std) * float(noise_std_scale)
        noise_stds = rng.uniform(lo, hi, size=M)

    # --- Build channels ---
    # Channel with lag tau takes mother[max_lag - tau : max_lag - tau + T],
    # i.e., it observes the mother tau steps in the past relative to lag-0 channels.
    data = np.empty((T, M))
    for i, tau in enumerate(lags):
        offset = max_lag - tau
        data[:, i] = mother[offset : offset + T] + rng.normal(0, noise_stds[i], T)

    result = _maybe_zscore(data, zscore=zscore)
    if return_internals:
        return result, LaggedInternals(mother=mother, lags=lags)
    return result


@dataclass
class SmoothSinInternals:
    """Internals returned by generate_sin_mts_smooth when return_internals=True."""
    a_values: np.ndarray  # (M,) sampled half-widths, one per channel


_SIN_INTERVALS = {
    "linear":        (-np.pi / 16, np.pi / 16), # for a \in [32, 16, 8, 4] = [0.16%, 0.65%, 2.6%, 11%] max deviation from linear (| sinx - x| )
    "monotonic":     (-np.pi / 2,  np.pi / 2),
    "non-monotonic": (-np.pi,       np.pi),
}


def _sin_channels(data: np.ndarray, channel_regimes: list, T: int, noise_stds, rng) -> None:
    """Fill data[:, i] in-place for each channel given its regime.

    noise_stds: scalar or array-like of length len(channel_regimes).
    """
    noise_stds = np.broadcast_to(
        np.asarray(noise_stds, dtype=float), (len(channel_regimes),)
    )
    for i, reg in enumerate(channel_regimes):
        lo, hi = _SIN_INTERVALS[reg]
        t = np.linspace(lo, hi, T)
        data[:, i] = np.sin(t) + rng.normal(0, float(noise_stds[i]), T)


def generate_sin_mts(
    M: int,
    T: int,
    regime: str = "fixed",
    n_linear: int | None = None,
    n_monotonic: int | None = None,
    noise_std: float = 0.05,
    noise_std_variable: bool = False,
    noise_std_scale: float = np.e,
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
    noise_stds = _resolve_channel_noise_stds(
        noise_std, M, noise_std_variable=noise_std_variable,
        noise_std_scale=noise_std_scale, rng=rng,
    )
    data = np.empty((T, M))
    _sin_channels(data, channel_regimes, T, noise_stds, rng)
    return _maybe_zscore(data, zscore=zscore)


def generate_sin_mts_smooth(
    M: int,
    T: int,
    A: float = np.pi,
    k: float = np.pi,
    linspace: bool = False,
    noise_std: float = 0.05,
    noise_std_variable: bool = False,
    noise_std_scale: float = np.e,
    ar1_a: float = 0.8,
    ar1_noise_std: float = 1.0,
    zscore: bool = False,
    rng=None,
    mother: np.ndarray | None = None,
    return_internals: bool = False,
) -> np.ndarray | tuple[np.ndarray, SmoothSinInternals]:
    """
    M sinusoidal channels derived from a shared AR(1) mother signal,
    with per-channel interval half-width a_i ~ Uniform(pi/64, A).

    As A increases from pi/64 toward pi, the proportion of channels filtered
    through nonlinear (and eventually non-monotonic) transformations increases,
    smoothly interpolating across regime space rather than using discrete
    linear/monotonic/non-monotonic assignments.

    Per-channel construction:
        y_i = 2*a_i * m_bar - a_i                              # [0,1] -> [-a_i, a_i]
        g_i = tanh(k*sin(y_i)) / tanh(k*sin(min(a_i, pi/2)))  # normalised to [-1, 1]
        X^(i) = g_i + noise

    Approximate regime boundaries for a_i (with default k=pi):
        a_i < pi/32  : super-linear  (~0.16% max |sin(x)-x| deviation)
        pi/32 - pi/16: approximately linear  (~0.65% max deviation)
        pi/16 - pi/2 : nonlinear monotonic
        pi/2  - pi   : non-monotonic  (sin(y) cycles through extremum)

    Parameters
    ----------
    M             : number of channels
    T             : time steps
    A             : upper bound of Uniform(pi/64, A) for per-channel half-width.
                    A > pi/2 is needed to sample non-monotonic channels.
    k             : tanh compositional gain in tanh(k·sin(y)); default np.pi.
                    Larger k → stronger saturation near extremes (more step-like).
                    Does not change monotonicity structure; increase A for that.
    linspace      : if True, a_values = np.linspace(pi/64, A, M) (deterministic, evenly
                    spaced); if False (default), a_values ~ Uniform(pi/64, A).
    noise_std     : per-channel i.i.d. noise std
    ar1_a         : AR(1) autoregressive coefficient for mother signal
    ar1_noise_std : driving noise std of the mother signal
    zscore        : z-score output
    rng           : RNG instance or int seed
    mother        : optional 1D array of length T to use instead of AR(1)
    return_internals : if True, return (data, SmoothSinInternals)

    Returns shape (T, M), or tuple of ((T, M), SmoothSinInternals).
    """
    a_min = np.pi / 64
    if A <= a_min:
        raise ValueError(f"A must be > pi/64 ({a_min:.4f}), got {A}.")
    rng = _resolve_rng(None, rng)

    if mother is not None:
        mother = np.asarray(mother, dtype=float)
        if mother.ndim != 1 or len(mother) != T:
            raise ValueError(f"mother must be 1D of length T={T}, got shape {mother.shape}")
    else:
        m = np.zeros(T)
        for t in range(1, T):
            m[t] = ar1_a * m[t - 1] + rng.normal(0, ar1_noise_std)
        mother = m

    lo, hi = mother.min(), mother.max()
    m_bar = np.zeros(T) if hi == lo else (mother - lo) / (hi - lo)

    a_values = np.linspace(a_min, A, M) if linspace else rng.uniform(a_min, A, size=M)
    noise_stds = _resolve_channel_noise_stds(
        noise_std, M, noise_std_variable=noise_std_variable,
        noise_std_scale=noise_std_scale, rng=rng,
    )
    data = np.empty((T, M))
    for i, a_i in enumerate(a_values):
        y = 2 * a_i * m_bar - a_i
        # g = np.sin(y)        # pure sin (no tanh compression)
        # g_max = np.sin(min(a_i, np.pi / 2))
        g = np.tanh(k * np.sin(y))  # compositional: tanh(k·sin(y))
        g_max = np.tanh(k * np.sin(min(a_i, np.pi / 2)))
        data[:, i] = g / g_max + rng.normal(0, noise_stds[i], T)

    result = _maybe_zscore(data, zscore=zscore)
    if return_internals:
        return result, SmoothSinInternals(a_values=a_values)
    return result


def generate_sin_mts_mother(
    M: int,
    T: int,
    regime: str = "fixed",
    n_linear: int | None = None,
    n_monotonic: int | None = None,
    noise_std: float = 0.05,
    noise_std_variable: bool = False,
    noise_std_scale: float = np.e,
    ar1_a: float = 0.8,
    ar1_noise_std: float = 1.0,
    zscore: bool = False,
    _regime_seed: int | None = None,
    rng=None,
    mother: np.ndarray | None = None,
) -> np.ndarray:
    """
    M sinusoidal channels derived from a shared AR(1) mother signal.

    The mother is min-max scaled to [0,1], then each channel applies
    g(x) = sin((b-a)*x + a) for its regime-specific interval [a,b]:
      linear:        [a,b] = [-pi/16, pi/16]  — approximately linear
      monotonic:     [a,b] = [-pi/2,  pi/2]   — strictly increasing
      non-monotonic: [a,b] = [-pi,    pi]     — full cycle

    Channel assignment:
      regime='fixed':  first n_linear channels linear, next n_monotonic monotonic,
                       remaining non-monotonic. Defaults: M//3 each.
      regime='random': each channel randomly assigned (uniform over 3 regimes).
                       Pass _regime_seed for reproducible assignment independent of noise seed.

    Parameters
    ----------
    M             : number of channels
    T             : time steps
    regime        : 'fixed' | 'random'
    n_linear      : linear channels (fixed only; default M//3)
    n_monotonic   : monotonic channels (fixed only; default M//3)
    noise_std     : per-channel i.i.d. noise std added after transformation
    ar1_a         : AR(1) autoregressive coefficient for mother signal
    ar1_noise_std : driving noise std of the mother signal
    zscore        : z-score output
    _regime_seed  : seed for random regime assignment (random regime only)
    rng           : RNG instance or int seed
    mother        : optional 1D array of length T to use as mother instead of AR(1)

    Returns shape (T, M).
    Slug convention: M<M>_T<T>_I<I>_l-<n_linear>_m-<n_monotonic>_nm-<n_nonmonotonic>
    """
    rng = _resolve_rng(None, rng)

    # Mother signal
    if mother is not None:
        mother = np.asarray(mother, dtype=float)
        if mother.ndim != 1 or len(mother) != T:
            raise ValueError(f"mother must be 1D of length T={T}, got shape {mother.shape}")
    else:
        m = np.zeros(T)
        for t in range(1, T):
            m[t] = ar1_a * m[t - 1] + rng.normal(0, ar1_noise_std)
        mother = m

    # Min-max scale to [0, 1]
    lo, hi = mother.min(), mother.max()
    m_bar = np.zeros(T) if hi == lo else (mother - lo) / (hi - lo)

    # Channel regime assignment
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

    noise_stds = _resolve_channel_noise_stds(
        noise_std, M, noise_std_variable=noise_std_variable,
        noise_std_scale=noise_std_scale, rng=rng,
    )
    data = np.empty((T, M))
    for i, reg in enumerate(channel_regimes):
        a, b = _SIN_INTERVALS[reg]
        y = (b - a) * m_bar + a
        g = np.sin(y)
        g_max = np.sin(min(abs(b), np.pi / 2))
        data[:, i] = g / g_max + rng.normal(0, noise_stds[i], T)

    return _maybe_zscore(data, zscore=zscore)

