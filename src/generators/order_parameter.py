from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from ..cml_order_parameter import summarize_field
from ._common import _maybe_zscore, _resolve_rng


@dataclass
class KuramotoOrderInternals:
    full_phases: np.ndarray | None
    r_full: np.ndarray
    r_observed: np.ndarray
    r_unobserved: np.ndarray
    r_full_future: np.ndarray
    r_unobserved_future: np.ndarray
    frequencies: np.ndarray
    observation_indices: np.ndarray
    sensor_offsets: np.ndarray
    initial_phases: np.ndarray
    final_phases: np.ndarray
    critical_coupling: float


def kuramoto_critical_coupling(
    frequency_distribution: str,
    omega_std: float = 1.0,
) -> float:
    """Continuum onset ``2 / (pi g(0))`` for the supported distributions."""

    distribution = frequency_distribution.strip().lower()
    scale = float(omega_std)
    if scale <= 0:
        raise ValueError(f"omega_std must be positive, got {omega_std}")
    if distribution == "gaussian":
        return float(np.sqrt(8.0 / np.pi) * scale)
    if distribution == "logistic":
        return float(8.0 * np.sqrt(3.0) * scale / np.pi**2)
    raise ValueError(
        f"unsupported frequency_distribution {frequency_distribution!r}; "
        "expected 'gaussian' or 'logistic'"
    )


def _sample_standardized_frequencies(
    *,
    size: int,
    distribution: str,
    sampling: str,
    rng,
) -> np.ndarray:
    distribution = distribution.strip().lower()
    sampling = sampling.strip().lower()
    if distribution not in {"gaussian", "logistic"}:
        raise ValueError(
            f"unsupported frequency_distribution {distribution!r}; "
            "expected 'gaussian' or 'logistic'"
        )
    if sampling not in {"random", "regular"}:
        raise ValueError(
            f"unsupported frequency_sampling {sampling!r}; expected 'random' or 'regular'"
        )

    if sampling == "random":
        if distribution == "gaussian":
            values = rng.standard_normal(size)
        else:
            values = rng.logistic(0.0, np.sqrt(3.0) / np.pi, size=size)
    else:
        probabilities = (np.arange(size, dtype=float) + 0.5) / size
        if distribution == "gaussian":
            values = ndtri(probabilities)
        else:
            values = (np.sqrt(3.0) / np.pi) * np.log(
                probabilities / (1.0 - probabilities)
            )
        values = values[rng.permutation(size)]
    return np.asarray(values, dtype=np.float64)


def _kuramoto_rhs(theta: np.ndarray, frequencies: np.ndarray, coupling: float) -> np.ndarray:
    sine = np.sin(theta)
    cosine = np.cos(theta)
    mean_sine = float(sine.mean())
    mean_cosine = float(cosine.mean())
    return frequencies + coupling * (mean_sine * cosine - mean_cosine * sine)


def _kuramoto_rk4_step(
    theta: np.ndarray,
    frequencies: np.ndarray,
    coupling: float,
    dt: float,
) -> np.ndarray:
    k1 = _kuramoto_rhs(theta, frequencies, coupling)
    k2 = _kuramoto_rhs(theta + 0.5 * dt * k1, frequencies, coupling)
    k3 = _kuramoto_rhs(theta + 0.5 * dt * k2, frequencies, coupling)
    k4 = _kuramoto_rhs(theta + dt * k3, frequencies, coupling)
    return np.mod(theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 2.0 * np.pi)


def _phase_coherence(phases: np.ndarray) -> np.ndarray:
    return np.abs(np.mean(np.exp(1j * phases), axis=-1))


def generate_kuramoto_order_parameter(
    M: int,
    T: int,
    K: float = np.sqrt(8.0 / np.pi),
    N_full: int = 256,
    dt: float = 0.02,
    sample_dt: float = 0.1,
    burn_time: float = 100.0,
    omega_mean: float = 1.0,
    omega_std: float = 1.0,
    frequency_distribution: str = "gaussian",
    frequency_sampling: str = "random",
    output: str = "cos",
    sensor_phase_std: float = 0.0,
    future_truth_T: int = 0,
    rng=None,
    zscore: bool = False,
    return_internals: bool = False,
    store_full_phases: bool = True,
):
    """Canonical all-to-all Kuramoto benchmark with hidden full-system truth.

    ``M`` channels are a nested random subset of a separately simulated
    ``N_full`` population. The returned MTS never contains ``R`` or the hidden
    phases. Reusing the same RNG seed across coupling values keeps frequencies,
    initial phases, observation indices and sensor offsets paired.
    """

    rng = _resolve_rng(None, rng)
    M = int(M)
    T = int(T)
    N_full = int(N_full)
    dt = float(dt)
    sample_dt = float(sample_dt)
    burn_time = float(burn_time)
    coupling = float(K)
    sensor_phase_std = float(sensor_phase_std)
    future_truth_T = int(future_truth_T)
    if M <= 0 or T <= 0 or N_full < M or future_truth_T < 0:
        raise ValueError(f"require 0 < M <= N_full and T > 0; got M={M}, N_full={N_full}, T={T}")
    if dt <= 0 or sample_dt < dt or burn_time < 0:
        raise ValueError(
            f"require dt > 0, sample_dt >= dt and burn_time >= 0; "
            f"got dt={dt}, sample_dt={sample_dt}, burn_time={burn_time}"
        )
    sample_every = int(round(sample_dt / dt))
    if not np.isclose(sample_every * dt, sample_dt, rtol=0.0, atol=1e-12):
        raise ValueError(f"sample_dt={sample_dt} must be an integer multiple of dt={dt}")
    burn_steps = int(round(burn_time / dt))
    if not np.isclose(burn_steps * dt, burn_time, rtol=0.0, atol=1e-12):
        raise ValueError(f"burn_time={burn_time} must be an integer multiple of dt={dt}")
    if sensor_phase_std < 0:
        raise ValueError(f"sensor_phase_std must be non-negative, got {sensor_phase_std}")

    standardized = _sample_standardized_frequencies(
        size=N_full,
        distribution=frequency_distribution,
        sampling=frequency_sampling,
        rng=rng,
    )
    frequencies = float(omega_mean) + float(omega_std) * standardized
    theta = rng.uniform(0.0, 2.0 * np.pi, size=N_full)
    initial_phases = theta.copy()
    observation_indices = rng.permutation(N_full)[:M]
    sensor_offsets_full = (
        rng.normal(0.0, sensor_phase_std, size=N_full)
        if sensor_phase_std
        else np.zeros(N_full, dtype=np.float64)
    )

    for _ in range(burn_steps):
        theta = _kuramoto_rk4_step(theta, frequencies, coupling, dt)

    phases = np.empty((T + future_truth_T, N_full), dtype=np.float64)
    for sample in range(T + future_truth_T):
        for _ in range(sample_every):
            theta = _kuramoto_rk4_step(theta, frequencies, coupling, dt)
        phases[sample] = theta

    current_phases = phases[:T]
    future_phases = phases[T:]
    observed_phase = current_phases[:, observation_indices]
    sensor_offsets = sensor_offsets_full[observation_indices]
    measured_phase = observed_phase + sensor_offsets[None, :]
    output_key = output.strip().lower()
    if output_key == "cos":
        observed = np.cos(measured_phase)
    elif output_key == "sin":
        observed = np.sin(measured_phase)
    elif output_key == "phase":
        observed = np.mod(measured_phase, 2.0 * np.pi)
    else:
        raise ValueError(f"unsupported output {output!r}; expected 'cos', 'sin' or 'phase'")
    observed = _maybe_zscore(observed, zscore=zscore)

    if not return_internals:
        return observed
    hidden_indices = np.setdiff1d(np.arange(N_full), observation_indices)
    internals = KuramotoOrderInternals(
        full_phases=current_phases if store_full_phases else None,
        r_full=_phase_coherence(current_phases),
        r_observed=_phase_coherence(observed_phase),
        r_unobserved=(
            _phase_coherence(current_phases[:, hidden_indices])
            if hidden_indices.size
            else np.full(T, np.nan)
        ),
        r_full_future=_phase_coherence(future_phases),
        r_unobserved_future=(
            _phase_coherence(future_phases[:, hidden_indices])
            if hidden_indices.size
            else np.full(future_truth_T, np.nan)
        ),
        frequencies=frequencies,
        observation_indices=observation_indices,
        sensor_offsets=sensor_offsets,
        initial_phases=initial_phases,
        final_phases=theta.copy(),
        critical_coupling=kuramoto_critical_coupling(
            frequency_distribution, omega_std=float(omega_std)
        ),
    )
    return observed, internals


@dataclass
class MillerHuseInternals:
    full_field: np.ndarray | None
    magnetization: np.ndarray
    spin_magnetization: np.ndarray
    spin_magnetization_unobserved: np.ndarray
    magnetization_future: np.ndarray
    spin_magnetization_future: np.ndarray
    spin_magnetization_unobserved_future: np.ndarray
    patch_indices: np.ndarray
    initial_field: np.ndarray
    final_field: np.ndarray


@dataclass
class StuartLandauInternals:
    full_states: np.ndarray | None
    order_parameter: np.ndarray
    mean_activity: np.ndarray
    order_parameter_future: np.ndarray
    mean_activity_future: np.ndarray
    frequencies: np.ndarray
    observation_indices: np.ndarray
    initial_state: np.ndarray
    final_state: np.ndarray


@dataclass
class QuadraticCMLInternals:
    truth_summary: dict[str, object]
    observation_indices: np.ndarray
    final_state: np.ndarray
    truth_field: np.ndarray | None


@dataclass
class DesaiZwanzigInternals:
    full_states: np.ndarray | None
    mean_field: np.ndarray
    mean_field_future: np.ndarray
    observation_indices: np.ndarray
    initial_state: np.ndarray
    final_state: np.ndarray


DESAI_ZWANZIG_REFERENCE_SIGMA_C = 1.890


def miller_huse_map(values: np.ndarray, mu: float = 3.0) -> np.ndarray:
    """Odd piecewise-linear Miller--Huse family on ``[-1, 1]``."""

    x = np.asarray(values, dtype=np.float64)
    slope = float(mu)
    if not 0.0 < slope <= 3.0:
        raise ValueError(f"mu must lie in (0, 3], got {mu}")
    return np.where(
        x < -1.0 / 3.0,
        -2.0 * slope / 3.0 - slope * x,
        np.where(
            x <= 1.0 / 3.0,
            slope * x,
            2.0 * slope / 3.0 - slope * x,
        ),
    )


def _rectangular_patch_shape(
    M: int,
    patch_shape: tuple[int, int] | list[int] | None,
) -> tuple[int, int]:
    if patch_shape is not None:
        if len(patch_shape) != 2:
            raise ValueError(f"patch_shape must contain two dimensions, got {patch_shape}")
        height, width = (int(value) for value in patch_shape)
        if height <= 0 or width <= 0 or height * width != M:
            raise ValueError(f"patch_shape={patch_shape} must have positive area M={M}")
        return height, width
    height = max(divisor for divisor in range(1, int(np.sqrt(M)) + 1) if M % divisor == 0)
    return height, M // height


def _spin_field(values: np.ndarray) -> np.ndarray:
    """Binary spin convention; exact zeros belong to the positive phase."""

    return np.where(values >= 0.0, 1.0, -1.0)


def generate_miller_huse(
    M: int,
    T: int,
    coupling: float = 0.205,
    mu: float = 3.0,
    lattice_side: int = 64,
    transients: int = 100_000,
    sample_every: int = 1,
    future_truth_T: int = 0,
    truth_start_T: int | None = None,
    patch_shape: tuple[int, int] | list[int] | None = None,
    patch_row: int | None = None,
    patch_col: int | None = None,
    observation_mode: str = "patch",
    initial_state: str = "random",
    rng=None,
    zscore: bool = False,
    return_internals: bool = False,
    store_full_field: bool = False,
):
    """Two-dimensional Miller--Huse CML with hidden global future truth."""

    rng = _resolve_rng(None, rng)
    M = int(M)
    T = int(T)
    side = int(lattice_side)
    transients = int(transients)
    sample_every = int(sample_every)
    future_truth_T = int(future_truth_T)
    truth_start = T if truth_start_T is None else int(truth_start_T)
    observation_key = str(observation_mode).strip().lower()
    if observation_key not in {"patch", "distributed"}:
        raise ValueError(
            f"unsupported observation_mode {observation_mode!r}; expected patch or distributed"
        )
    if observation_key == "patch":
        patch_height, patch_width = _rectangular_patch_shape(M, patch_shape)
    else:
        patch_height, patch_width = 1, M
    if (
        T <= 0
        or (observation_key == "patch" and side < max(patch_height, patch_width))
        or (observation_key == "distributed" and side * side < M)
        or transients < 0
        or sample_every <= 0
        or future_truth_T < 0
        or truth_start < T
    ):
        raise ValueError(
            f"invalid sizes: M={M}, T={T}, lattice_side={side}, "
            f"transients={transients}, sample_every={sample_every}, "
            f"future_truth_T={future_truth_T}"
        )
    g = float(coupling)
    if not 0.0 <= g <= 0.25:
        raise ValueError(f"coupling must lie in [0, .25] for convex four-neighbour mixing, got {g}")

    initial_key = str(initial_state).strip().lower()
    if initial_key == "random":
        state = rng.uniform(-1.0, 1.0, size=(side, side))
    elif initial_key in {"positive", "ordered_positive"}:
        state = rng.uniform(0.0, 1.0, size=(side, side))
    elif initial_key in {"negative", "ordered_negative"}:
        state = rng.uniform(-1.0, 0.0, size=(side, side))
    else:
        raise ValueError(
            f"unsupported initial_state {initial_state!r}; expected random, positive or negative"
        )
    initial_field = state.copy()
    if observation_key == "patch":
        row = int(rng.integers(side)) if patch_row is None else int(patch_row) % side
        col = int(rng.integers(side)) if patch_col is None else int(patch_col) % side
        rows = (row + np.arange(patch_height)) % side
        cols = (col + np.arange(patch_width)) % side
        patch_indices = np.array(np.meshgrid(rows, cols, indexing="ij")).reshape(2, -1).T
    else:
        flat_indices = rng.permutation(side * side)[:M]
        patch_indices = np.column_stack(np.divmod(flat_indices, side))

    def step(field: np.ndarray) -> np.ndarray:
        mapped = miller_huse_map(field, mu=mu)
        neighbours = (
            np.roll(mapped, 1, axis=0)
            + np.roll(mapped, -1, axis=0)
            + np.roll(mapped, 1, axis=1)
            + np.roll(mapped, -1, axis=1)
        )
        return (1.0 - 4.0 * g) * mapped + g * neighbours

    for _ in range(transients):
        state = step(state)

    total_samples = truth_start + future_truth_T
    observed = np.empty((T, M), dtype=np.float64)
    full_field = np.empty((T, side, side), dtype=np.float64) if store_full_field else None
    magnetization = np.empty(T, dtype=np.float64)
    spin_magnetization = np.empty(T, dtype=np.float64)
    spin_magnetization_unobserved = np.empty(T, dtype=np.float64)
    magnetization_future = np.empty(future_truth_T, dtype=np.float64)
    spin_magnetization_future = np.empty(future_truth_T, dtype=np.float64)
    spin_magnetization_unobserved_future = np.empty(future_truth_T, dtype=np.float64)
    hidden_count = side * side - M
    for sample in range(total_samples):
        for _ in range(sample_every):
            state = step(state)
        patch = state[patch_indices[:, 0], patch_indices[:, 1]]
        spins = _spin_field(state)
        spin_mean = float(spins.mean())
        hidden_spin_mean = (
            float((spins.sum() - _spin_field(patch).sum()) / hidden_count)
            if hidden_count
            else np.nan
        )
        if sample < T:
            observed[sample] = patch
            magnetization[sample] = float(state.mean())
            spin_magnetization[sample] = spin_mean
            spin_magnetization_unobserved[sample] = hidden_spin_mean
            if full_field is not None:
                full_field[sample] = state
        elif sample >= truth_start:
            future_index = sample - truth_start
            magnetization_future[future_index] = float(state.mean())
            spin_magnetization_future[future_index] = spin_mean
            spin_magnetization_unobserved_future[future_index] = hidden_spin_mean

    observed = _maybe_zscore(observed, zscore=zscore)
    if not return_internals:
        return observed
    return observed, MillerHuseInternals(
        full_field=full_field,
        magnetization=magnetization,
        spin_magnetization=spin_magnetization,
        spin_magnetization_unobserved=spin_magnetization_unobserved,
        magnetization_future=magnetization_future,
        spin_magnetization_future=spin_magnetization_future,
        spin_magnetization_unobserved_future=spin_magnetization_unobserved_future,
        patch_indices=patch_indices,
        initial_field=initial_field,
        final_field=state.copy(),
    )


def generate_desai_zwanzig(
    M: int,
    T: int,
    sigma: float = DESAI_ZWANZIG_REFERENCE_SIGMA_C,
    N_full: int | None = None,
    alpha: float = 1.0,
    theta: float = 4.0,
    sigma_m: float = 0.8,
    nu: float = 0.5,
    dt: float = 0.005,
    sample_dt: float = 0.05,
    burn_time: float = 100.0,
    future_truth_T: int = 0,
    truth_start_T: int | None = None,
    initial_mean: float = 1.0,
    initial_std: float = 0.1,
    rng=None,
    zscore: bool = False,
    return_internals: bool = False,
    store_full_states: bool = False,
):
    r"""Desai--Zwanzig interacting diffusions with multiplicative noise.

    This is Eq. (1) of Evangelou et al. (PRE 110, 014121, 2024),

    .. math::

       dx_i = [-x_i^3 + (\alpha + \nu\sigma_m^2)x_i
               - \theta(x_i-\bar{x})]dt
              + \sqrt{\sigma^2+\sigma_m^2x_i^2}\,dW_i.

    Their canonical parameters are ``alpha=1``, ``theta=4``,
    ``sigma_m=0.8`` and ``nu=1/2``. In the mean-field limit the first moment
    ``M1=mean_i(x_i)`` undergoes a continuous pitchfork at
    ``sigma ~= 1.890``. Finite systems can switch between the two branches,
    so the benchmark scalar is the time mean of ``abs(M1)`` on a disjoint
    future window.
    """

    rng = _resolve_rng(None, rng)
    M = int(M)
    T = int(T)
    N = M if N_full is None else int(N_full)
    future_truth_T = int(future_truth_T)
    truth_start = T if truth_start_T is None else int(truth_start_T)
    sigma = float(sigma)
    sigma_m = float(sigma_m)
    dt = float(dt)
    sample_dt = float(sample_dt)
    burn_time = float(burn_time)
    initial_std = float(initial_std)
    if M <= 0 or T <= 0 or N < M or future_truth_T < 0 or truth_start < T:
        raise ValueError(
            f"require 0 < M <= N_full, T > 0 and truth_start_T >= T; "
            f"got M={M}, N_full={N}, T={T}, truth_start_T={truth_start}"
        )
    if sigma < 0.0 or sigma_m < 0.0 or float(theta) < 0.0 or initial_std < 0.0:
        raise ValueError(
            "sigma, sigma_m, theta and initial_std must be non-negative; "
            f"got {sigma}, {sigma_m}, {theta}, {initial_std}"
        )
    if dt <= 0.0 or sample_dt < dt or burn_time < 0.0:
        raise ValueError(
            f"require dt > 0, sample_dt >= dt and burn_time >= 0; "
            f"got dt={dt}, sample_dt={sample_dt}, burn_time={burn_time}"
        )
    sample_every = int(round(sample_dt / dt))
    burn_steps = int(round(burn_time / dt))
    if not np.isclose(sample_every * dt, sample_dt, rtol=0.0, atol=1e-12):
        raise ValueError(f"sample_dt={sample_dt} must be an integer multiple of dt={dt}")
    if not np.isclose(burn_steps * dt, burn_time, rtol=0.0, atol=1e-12):
        raise ValueError(f"burn_time={burn_time} must be an integer multiple of dt={dt}")

    state = float(initial_mean) + initial_std * rng.standard_normal(N)
    initial_state = state.copy()
    observation_indices = rng.permutation(N)[:M]
    noise_scale = np.sqrt(dt)
    linear = float(alpha) + float(nu) * sigma_m**2
    coupling = float(theta)

    def step(values: np.ndarray) -> np.ndarray:
        mean_field = float(values.mean())
        drift = -values**3 + linear * values - coupling * (values - mean_field)
        diffusion = np.sqrt(sigma**2 + sigma_m**2 * values**2)
        updated = values + drift * dt + diffusion * noise_scale * rng.standard_normal(N)
        if not np.isfinite(updated).all():
            raise FloatingPointError(
                "Desai--Zwanzig Euler--Maruyama path became non-finite; "
                "reduce dt or change the seed"
            )
        return updated

    for _ in range(burn_steps):
        state = step(state)

    observed = np.empty((T, M), dtype=np.float64)
    mean_field = np.empty(T, dtype=np.float64)
    mean_field_future = np.empty(future_truth_T, dtype=np.float64)
    full_states = (
        np.empty((T, N), dtype=np.float64) if store_full_states else None
    )
    for sample in range(truth_start + future_truth_T):
        for _ in range(sample_every):
            state = step(state)
        if sample < T:
            observed[sample] = state[observation_indices]
            mean_field[sample] = float(state.mean())
            if full_states is not None:
                full_states[sample] = state
        if sample >= truth_start:
            mean_field_future[sample - truth_start] = float(state.mean())

    observed = _maybe_zscore(observed, zscore=zscore)
    if not return_internals:
        return observed
    return observed, DesaiZwanzigInternals(
        full_states=full_states,
        mean_field=mean_field,
        mean_field_future=mean_field_future,
        observation_indices=observation_indices,
        initial_state=initial_state,
        final_state=state.copy(),
    )


def _stuart_landau_rhs(
    state: np.ndarray,
    frequencies: np.ndarray,
    coupling: float,
) -> np.ndarray:
    mean_field = np.mean(state)
    return (
        1.0 + 1j * frequencies - np.abs(state) ** 2
    ) * state + coupling * (mean_field - state)


def _stuart_landau_step(
    state: np.ndarray,
    frequencies: np.ndarray,
    coupling: float,
    dt: float,
) -> np.ndarray:
    k1 = _stuart_landau_rhs(state, frequencies, coupling)
    k2 = _stuart_landau_rhs(state + 0.5 * dt * k1, frequencies, coupling)
    k3 = _stuart_landau_rhs(state + 0.5 * dt * k2, frequencies, coupling)
    k4 = _stuart_landau_rhs(state + dt * k3, frequencies, coupling)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def generate_stuart_landau(
    M: int,
    T: int,
    coupling: float = 0.8,
    frequency_half_width: float = 0.8,
    N_full: int | None = None,
    omega_mean: float = 2.0,
    dt: float = 0.02,
    sample_dt: float = 0.1,
    burn_time: float = 200.0,
    future_truth_T: int = 0,
    truth_start_T: int | None = None,
    output: str = "real",
    rng=None,
    zscore: bool = False,
    return_internals: bool = False,
    store_full_states: bool = False,
):
    """Mean-field Stuart--Landau population from Matthews--Strogatz (1990).

    Relative frequencies are evenly spaced on ``[-gamma, gamma]`` and randomly
    assigned to oscillators. ``omega_mean`` restores a laboratory-frame carrier;
    it changes only the common rotation and leaves the published phase diagram
    in ``(coupling, frequency_half_width)`` unchanged.
    """

    rng = _resolve_rng(None, rng)
    M = int(M)
    T = int(T)
    N = M if N_full is None else int(N_full)
    future_truth_T = int(future_truth_T)
    truth_start = T if truth_start_T is None else int(truth_start_T)
    coupling = float(coupling)
    gamma = float(frequency_half_width)
    dt = float(dt)
    sample_dt = float(sample_dt)
    burn_time = float(burn_time)
    if M <= 0 or T <= 0 or N < M or future_truth_T < 0 or truth_start < T:
        raise ValueError(f"require 0 < M <= N_full and T > 0; got M={M}, N_full={N}, T={T}")
    if coupling < 0.0 or gamma < 0.0:
        raise ValueError(
            f"coupling and frequency_half_width must be non-negative, got {coupling}, {gamma}"
        )
    if dt <= 0.0 or sample_dt < dt or burn_time < 0.0:
        raise ValueError(
            f"require dt > 0, sample_dt >= dt and burn_time >= 0; "
            f"got dt={dt}, sample_dt={sample_dt}, burn_time={burn_time}"
        )
    sample_every = int(round(sample_dt / dt))
    burn_steps = int(round(burn_time / dt))
    if not np.isclose(sample_every * dt, sample_dt, rtol=0.0, atol=1e-12):
        raise ValueError(f"sample_dt={sample_dt} must be an integer multiple of dt={dt}")
    if not np.isclose(burn_steps * dt, burn_time, rtol=0.0, atol=1e-12):
        raise ValueError(f"burn_time={burn_time} must be an integer multiple of dt={dt}")

    relative = np.linspace(-gamma, gamma, N, dtype=np.float64)
    frequencies = float(omega_mean) + relative[rng.permutation(N)]
    state = rng.uniform(-1.0, 1.0, N) + 1j * rng.uniform(-1.0, 1.0, N)
    initial_state = state.copy()
    observation_indices = rng.permutation(N)[:M]
    for _ in range(burn_steps):
        state = _stuart_landau_step(state, frequencies, coupling, dt)

    total_samples = truth_start + future_truth_T
    states = np.empty((total_samples, N), dtype=np.complex128)
    for sample in range(total_samples):
        for _ in range(sample_every):
            state = _stuart_landau_step(state, frequencies, coupling, dt)
        states[sample] = state

    current = states[:T]
    future = states[truth_start:]
    observed_state = current[:, observation_indices]
    output_key = str(output).strip().lower()
    if output_key == "real":
        observed = observed_state.real
    elif output_key == "imag":
        observed = observed_state.imag
    elif output_key == "amplitude":
        observed = np.abs(observed_state)
    elif output_key == "phase":
        observed = np.angle(observed_state)
    else:
        raise ValueError(
            f"unsupported output {output!r}; expected real, imag, amplitude or phase"
        )
    observed = _maybe_zscore(observed, zscore=zscore)
    if not return_internals:
        return observed
    return observed, StuartLandauInternals(
        full_states=current if store_full_states else None,
        order_parameter=np.mean(current, axis=1),
        mean_activity=np.mean(np.abs(current) ** 2, axis=1),
        order_parameter_future=np.mean(future, axis=1),
        mean_activity_future=np.mean(np.abs(future) ** 2, axis=1),
        frequencies=frequencies,
        observation_indices=observation_indices,
        initial_state=initial_state,
        final_state=state.copy(),
    )


def generate_quadratic_cml_order_parameter(
    M: int,
    T: int,
    alpha: float = 1.8,
    eps: float = 0.3,
    lattice_size: int = 512,
    transients: int = 2_000_000,
    sample_every: int = 1,
    truth_start_T: int = 1000,
    future_truth_T: int = 20_000,
    observation_mode: str = "distributed",
    selected_spatial_band: tuple[float, float] = (0.25, 0.45),
    pattern_word_length: int = 4,
    rng=None,
    zscore: bool = False,
    return_internals: bool = False,
    store_truth_field: bool = False,
):
    """Quadratic CML with fixed-large-lattice observations and future truth.

    The MTS and full-field diagnostic window come from the same deterministic
    trajectory. ``truth_start_T`` fixes the future window across paired T
    prefixes, so every prefix has identical full-lattice physical truth.
    """

    from .dynamical import generate_cml_logistic

    rng = _resolve_rng(None, rng)
    M = int(M)
    T = int(T)
    lattice_size = int(lattice_size)
    truth_start_T = int(truth_start_T)
    future_truth_T = int(future_truth_T)
    if (
        M <= 0
        or T <= 0
        or lattice_size < M
        or truth_start_T < T
        or future_truth_T < 3
    ):
        raise ValueError(
            "require 0 < M <= lattice_size, truth_start_T >= T and "
            f"future_truth_T >= 3; got M={M}, T={T}, lattice_size={lattice_size}, "
            f"truth_start_T={truth_start_T}, future_truth_T={future_truth_T}"
        )
    generated = generate_cml_logistic(
        M=M,
        T=T,
        alpha=alpha,
        eps=eps,
        transients=transients,
        sample_every=sample_every,
        lattice_size=lattice_size,
        observation_mode=observation_mode,
        return_final_state=True,
        return_observation_indices=True,
        rng=rng,
        zscore=zscore,
    )
    observed, input_final_state, observation_indices = generated

    # The first record has time index zero and input_final_state has index T-1.
    # Skipping truth_start_T-T+1 updates therefore starts truth at the fixed
    # time index truth_start_T for every paired input prefix.
    truth_gap = (truth_start_T - T) * int(sample_every) + 1
    generated_truth = generate_cml_logistic(
        M=M,
        T=future_truth_T,
        alpha=alpha,
        eps=eps,
        transients=truth_gap,
        sample_every=sample_every,
        lattice_size=lattice_size,
        observation_mode=observation_mode,
        init_state=input_final_state,
        return_full_lattice=True,
        return_final_state=True,
        rng=rng,
        zscore=False,
    )
    _, truth_field, final_state = generated_truth
    summary = summarize_field(
        truth_field,
        max_spatial_lag=min(64, lattice_size // 2),
        selected_spatial_band=selected_spatial_band,
        pattern_word_length=pattern_word_length,
    )
    if not return_internals:
        return observed
    return observed, QuadraticCMLInternals(
        truth_summary=summary,
        observation_indices=np.asarray(observation_indices, dtype=np.int32),
        final_state=np.asarray(final_state, dtype=np.float64),
        truth_field=truth_field if store_truth_field else None,
    )
