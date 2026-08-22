from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri

from ._common import _maybe_zscore, _resolve_rng


@dataclass
class KuramotoOrderInternals:
    full_phases: np.ndarray | None
    r_full: np.ndarray
    r_observed: np.ndarray
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
    if M <= 0 or T <= 0 or N_full < M:
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

    phases = np.empty((T, N_full), dtype=np.float64)
    for sample in range(T):
        for _ in range(sample_every):
            theta = _kuramoto_rk4_step(theta, frequencies, coupling, dt)
        phases[sample] = theta

    observed_phase = phases[:, observation_indices]
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
    internals = KuramotoOrderInternals(
        full_phases=phases if store_full_phases else None,
        r_full=_phase_coherence(phases),
        r_observed=_phase_coherence(observed_phase),
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
    patch_indices: np.ndarray
    initial_field: np.ndarray
    final_field: np.ndarray


def miller_huse_map(values: np.ndarray) -> np.ndarray:
    """Original odd, piecewise-linear Miller--Huse local map on [-1, 1]."""

    x = np.asarray(values, dtype=np.float64)
    return np.where(
        x < -1.0 / 3.0,
        -2.0 - 3.0 * x,
        np.where(x < 1.0 / 3.0, 3.0 * x, 2.0 - 3.0 * x),
    )


def generate_miller_huse(
    M: int,
    T: int,
    coupling: float = 0.205,
    lattice_side: int = 64,
    transients: int = 100_000,
    sample_every: int = 1,
    patch_row: int | None = None,
    patch_col: int | None = None,
    rng=None,
    zscore: bool = False,
    return_internals: bool = False,
    store_full_field: bool = True,
):
    """Two-dimensional Miller--Huse CML with a contiguous square observation patch."""

    rng = _resolve_rng(None, rng)
    M = int(M)
    T = int(T)
    side = int(lattice_side)
    transients = int(transients)
    sample_every = int(sample_every)
    patch_side = int(round(np.sqrt(M)))
    if patch_side * patch_side != M:
        raise ValueError(f"M must be a perfect square for a 2-D patch, got {M}")
    if T <= 0 or side < patch_side or transients < 0 or sample_every <= 0:
        raise ValueError(
            f"invalid sizes: M={M}, T={T}, lattice_side={side}, "
            f"transients={transients}, sample_every={sample_every}"
        )
    g = float(coupling)
    if not 0.0 <= g <= 0.25:
        raise ValueError(f"coupling must lie in [0, .25] for convex four-neighbour mixing, got {g}")

    state = rng.uniform(-1.0, 1.0, size=(side, side))
    initial_field = state.copy()
    row = int(rng.integers(side)) if patch_row is None else int(patch_row) % side
    col = int(rng.integers(side)) if patch_col is None else int(patch_col) % side
    rows = (row + np.arange(patch_side)) % side
    cols = (col + np.arange(patch_side)) % side
    patch_indices = np.array(np.meshgrid(rows, cols, indexing="ij")).reshape(2, -1).T

    def step(field: np.ndarray) -> np.ndarray:
        mapped = miller_huse_map(field)
        neighbours = (
            np.roll(mapped, 1, axis=0)
            + np.roll(mapped, -1, axis=0)
            + np.roll(mapped, 1, axis=1)
            + np.roll(mapped, -1, axis=1)
        )
        return (1.0 - 4.0 * g) * mapped + g * neighbours

    for _ in range(transients):
        state = step(state)

    observed = np.empty((T, M), dtype=np.float64)
    full_field = np.empty((T, side, side), dtype=np.float64) if store_full_field else None
    magnetization = np.empty(T, dtype=np.float64)
    spin_magnetization = np.empty(T, dtype=np.float64)
    for sample in range(T):
        for _ in range(sample_every):
            state = step(state)
        observed[sample] = state[np.ix_(rows, cols)].reshape(-1)
        magnetization[sample] = float(state.mean())
        spin_magnetization[sample] = float(np.sign(state).mean())
        if full_field is not None:
            full_field[sample] = state

    observed = _maybe_zscore(observed, zscore=zscore)
    if not return_internals:
        return observed
    return observed, MillerHuseInternals(
        full_field=full_field,
        magnetization=magnetization,
        spin_magnetization=spin_magnetization,
        patch_indices=patch_indices,
        initial_field=initial_field,
        final_field=state.copy(),
    )
