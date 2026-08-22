from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtri

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
    patch_shape: tuple[int, int] | list[int] | None = None,
    patch_row: int | None = None,
    patch_col: int | None = None,
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
    patch_height, patch_width = _rectangular_patch_shape(M, patch_shape)
    if (
        T <= 0
        or side < max(patch_height, patch_width)
        or transients < 0
        or sample_every <= 0
        or future_truth_T < 0
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
    row = int(rng.integers(side)) if patch_row is None else int(patch_row) % side
    col = int(rng.integers(side)) if patch_col is None else int(patch_col) % side
    rows = (row + np.arange(patch_height)) % side
    cols = (col + np.arange(patch_width)) % side
    patch_indices = np.array(np.meshgrid(rows, cols, indexing="ij")).reshape(2, -1).T

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

    total_samples = T + future_truth_T
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
        patch = state[np.ix_(rows, cols)].reshape(-1)
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
        else:
            future_index = sample - T
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


@dataclass
class KineticIsingInternals:
    full_spins: np.ndarray | None
    magnetization: np.ndarray
    magnetization_unobserved: np.ndarray
    magnetization_future: np.ndarray
    magnetization_unobserved_future: np.ndarray
    patch_indices: np.ndarray
    initial_spins: np.ndarray
    final_spins: np.ndarray
    beta: float
    reduced_coupling: float
    exact_spontaneous_magnetization: float


def ising_reduced_coupling(beta: float, J_x: float = 1.0, J_y: float = 1.0) -> float:
    """Anisotropic square-Ising coordinate ``sinh(2 beta Jx)sinh(2 beta Jy)``."""

    beta = float(beta)
    J_x = float(J_x)
    J_y = float(J_y)
    if beta < 0.0 or J_x <= 0.0 or J_y <= 0.0:
        raise ValueError(f"require beta >= 0 and J_x,J_y > 0; got {beta}, {J_x}, {J_y}")
    return float(np.sinh(2.0 * beta * J_x) * np.sinh(2.0 * beta * J_y))


def ising_beta_from_reduced_coupling(
    reduced_coupling: float,
    J_x: float = 1.0,
    J_y: float = 1.0,
) -> float:
    """Solve the exact anisotropic-Ising reduced-coupling relation for beta."""

    target = float(reduced_coupling)
    if target < 0.0 or J_x <= 0.0 or J_y <= 0.0:
        raise ValueError(
            f"require reduced_coupling >= 0 and J_x,J_y > 0; got {target}, {J_x}, {J_y}"
        )
    if target == 0.0:
        return 0.0
    upper = 1.0
    while ising_reduced_coupling(upper, J_x, J_y) < target:
        upper *= 2.0
    return float(
        brentq(
            lambda value: ising_reduced_coupling(value, J_x, J_y) - target,
            0.0,
            upper,
        )
    )


def ising_exact_spontaneous_magnetization(reduced_coupling: float) -> float:
    """Yang's thermodynamic-limit magnetization for the anisotropic square Ising model."""

    value = float(reduced_coupling)
    if value <= 1.0:
        return 0.0
    return float((1.0 - value**-2.0) ** 0.125)


def _wolff_equilibrate(
    spins: np.ndarray,
    *,
    beta: float,
    J_x: float,
    J_y: float,
    equivalent_sweeps: int,
    rng,
) -> None:
    """In-place Wolff equilibration, counted by total flipped lattice volumes."""

    if equivalent_sweeps <= 0:
        return
    side = spins.shape[0]
    target_flips = equivalent_sweeps * spins.size
    flips = 0
    p_x = 1.0 - np.exp(-2.0 * beta * J_x)
    p_y = 1.0 - np.exp(-2.0 * beta * J_y)
    while flips < target_flips:
        start_row = int(rng.integers(side))
        start_col = int(rng.integers(side))
        phase = spins[start_row, start_col]
        members = [(start_row, start_col)]
        stack = [(start_row, start_col)]
        included = np.zeros_like(spins, dtype=bool)
        included[start_row, start_col] = True
        while stack:
            row, col = stack.pop()
            for d_row, d_col, probability in (
                (-1, 0, p_y),
                (1, 0, p_y),
                (0, -1, p_x),
                (0, 1, p_x),
            ):
                neighbour = ((row + d_row) % side, (col + d_col) % side)
                if (
                    not included[neighbour]
                    and spins[neighbour] == phase
                    and rng.random() < probability
                ):
                    included[neighbour] = True
                    members.append(neighbour)
                    stack.append(neighbour)
        member_rows, member_cols = np.asarray(members, dtype=np.int32).T
        spins[member_rows, member_cols] *= -1
        flips += len(members)


def generate_kinetic_ising(
    M: int,
    T: int,
    reduced_coupling: float | None = 1.0,
    beta: float | None = None,
    J_x: float = 1.0,
    J_y: float = 1.0,
    lattice_side: int = 64,
    equilibration_sweeps: int = 200,
    kinetic_burn_sweeps: int = 0,
    sample_every: int = 1,
    future_truth_T: int = 0,
    patch_shape: tuple[int, int] | list[int] | None = None,
    patch_row: int | None = None,
    patch_col: int | None = None,
    initial_state: str = "random",
    rng=None,
    zscore: bool = False,
    return_internals: bool = False,
    store_full_spins: bool = False,
):
    """Equilibrium anisotropic Ising field observed under checkerboard heat-bath dynamics."""

    rng = _resolve_rng(None, rng)
    M = int(M)
    T = int(T)
    side = int(lattice_side)
    equilibration_sweeps = int(equilibration_sweeps)
    kinetic_burn_sweeps = int(kinetic_burn_sweeps)
    sample_every = int(sample_every)
    future_truth_T = int(future_truth_T)
    patch_height, patch_width = _rectangular_patch_shape(M, patch_shape)
    if (
        T <= 0
        or side < max(patch_height, patch_width)
        or equilibration_sweeps < 0
        or kinetic_burn_sweeps < 0
        or sample_every <= 0
        or future_truth_T < 0
    ):
        raise ValueError(
            f"invalid sizes or sweep counts: M={M}, T={T}, lattice_side={side}, "
            f"equilibration_sweeps={equilibration_sweeps}, "
            f"kinetic_burn_sweeps={kinetic_burn_sweeps}, sample_every={sample_every}, "
            f"future_truth_T={future_truth_T}"
        )
    if beta is not None and reduced_coupling is not None:
        raise ValueError("specify either beta or reduced_coupling, not both")
    if beta is None:
        if reduced_coupling is None:
            raise ValueError("one of beta or reduced_coupling is required")
        beta_value = ising_beta_from_reduced_coupling(reduced_coupling, J_x, J_y)
    else:
        beta_value = float(beta)
    u_value = ising_reduced_coupling(beta_value, J_x, J_y)

    initial_key = str(initial_state).strip().lower()
    if initial_key == "random":
        spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=(side, side))
    elif initial_key in {"positive", "ordered_positive"}:
        spins = np.ones((side, side), dtype=np.int8)
    elif initial_key in {"negative", "ordered_negative"}:
        spins = -np.ones((side, side), dtype=np.int8)
    else:
        raise ValueError(
            f"unsupported initial_state {initial_state!r}; expected random, positive or negative"
        )
    initial_spins = spins.copy()
    row = int(rng.integers(side)) if patch_row is None else int(patch_row) % side
    col = int(rng.integers(side)) if patch_col is None else int(patch_col) % side
    rows = (row + np.arange(patch_height)) % side
    cols = (col + np.arange(patch_width)) % side
    patch_indices = np.array(np.meshgrid(rows, cols, indexing="ij")).reshape(2, -1).T

    _wolff_equilibrate(
        spins,
        beta=beta_value,
        J_x=float(J_x),
        J_y=float(J_y),
        equivalent_sweeps=equilibration_sweeps,
        rng=rng,
    )
    row_grid, col_grid = np.indices((side, side))
    checkerboards = ((row_grid + col_grid) % 2 == 0, (row_grid + col_grid) % 2 == 1)

    def heat_bath_sweep() -> None:
        for mask in checkerboards:
            local_field = float(J_x) * (
                np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1)
            ) + float(J_y) * (
                np.roll(spins, 1, axis=0) + np.roll(spins, -1, axis=0)
            )
            p_plus = 1.0 / (1.0 + np.exp(-2.0 * beta_value * local_field))
            draws = rng.random(spins.shape)
            spins[mask] = np.where(draws[mask] < p_plus[mask], 1, -1)

    for _ in range(kinetic_burn_sweeps):
        heat_bath_sweep()

    total_samples = T + future_truth_T
    observed = np.empty((T, M), dtype=np.float64)
    full_spins = np.empty((T, side, side), dtype=np.int8) if store_full_spins else None
    magnetization = np.empty(T, dtype=np.float64)
    magnetization_unobserved = np.empty(T, dtype=np.float64)
    magnetization_future = np.empty(future_truth_T, dtype=np.float64)
    magnetization_unobserved_future = np.empty(future_truth_T, dtype=np.float64)
    hidden_count = side * side - M
    for sample in range(total_samples):
        for _ in range(sample_every):
            heat_bath_sweep()
        patch = spins[np.ix_(rows, cols)].reshape(-1)
        mean = float(spins.mean())
        hidden_mean = (
            float((spins.sum() - patch.sum()) / hidden_count) if hidden_count else np.nan
        )
        if sample < T:
            observed[sample] = patch
            magnetization[sample] = mean
            magnetization_unobserved[sample] = hidden_mean
            if full_spins is not None:
                full_spins[sample] = spins
        else:
            future_index = sample - T
            magnetization_future[future_index] = mean
            magnetization_unobserved_future[future_index] = hidden_mean

    observed = _maybe_zscore(observed, zscore=zscore)
    if not return_internals:
        return observed
    return observed, KineticIsingInternals(
        full_spins=full_spins,
        magnetization=magnetization,
        magnetization_unobserved=magnetization_unobserved,
        magnetization_future=magnetization_future,
        magnetization_unobserved_future=magnetization_unobserved_future,
        patch_indices=patch_indices,
        initial_spins=initial_spins,
        final_spins=spins.copy(),
        beta=beta_value,
        reduced_coupling=u_value,
        exact_spontaneous_magnetization=ising_exact_spontaneous_magnetization(u_value),
    )
