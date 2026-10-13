"""SPI-independent diagnostics for the quadratic coupled-map lattice.

These quantities are intentionally computed from the physical field rather
than from SPI outputs.  They are candidates/validators for a macroscopic
transition coordinate, not labels used to learn the SPI--SPI embedding.
"""

from __future__ import annotations

import numpy as np


def _as_field(field: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 3:
        raise ValueError(f"field must have shape (T, L) with T,L >= 3, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("field contains non-finite values")
    return values


def period2_residual(field: np.ndarray) -> np.ndarray:
    """Absolute local departure from period-two recurrence, |x(t+2)-x(t)|."""

    values = _as_field(field)
    return np.abs(values[2:] - values[:-2])


def period2_activity(field: np.ndarray) -> float:
    """Threshold-free RMS loss of local period-two coherence."""

    residual = period2_residual(field)
    return float(np.sqrt(np.mean(residual * residual)))


def _normalised_entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    positive = values > 0
    if values.ndim != 1 or len(values) < 2 or not np.any(positive):
        return 0.0
    return float(-np.sum(values[positive] * np.log(values[positive])) / np.log(len(values)))


def spatial_power_distribution(field: np.ndarray) -> np.ndarray:
    """Time-averaged non-DC spatial power, normalised to sum to one.

    This is a full-ring diagnostic: callers using a cropped observation should
    not interpret it as a thermodynamic structure factor.
    """

    values = _as_field(field)
    centred = values - values.mean(axis=1, keepdims=True)
    power = np.mean(np.abs(np.fft.rfft(centred, axis=1)) ** 2, axis=0)[1:]
    total = float(power.sum())
    return np.zeros_like(power) if total <= 1e-15 else power / total


def spatial_spectral_entropy(field: np.ndarray) -> float:
    """Normalised Shannon entropy of the non-DC spatial power distribution."""

    return _normalised_entropy(spatial_power_distribution(field))


def spatial_spectral_concentration(field: np.ndarray) -> float:
    """Inverse-participation ratio of spatial power (one for a single mode)."""

    probabilities = spatial_power_distribution(field)
    return float(np.sum(probabilities * probabilities))


def spatial_order_magnitude(field: np.ndarray) -> float:
    """Finite-size-normalised magnitude of spatial spectral order.

    This is zero for power spread uniformly across all non-DC modes and one
    for a single coherent mode. It permits the selected wavelength to vary
    without tuning or selecting a mode separately at each alpha.
    """

    probabilities = spatial_power_distribution(field)
    mode_count = len(probabilities)
    if mode_count == 1:
        return 1.0 if probabilities[0] > 0 else 0.0
    concentration = float(np.sum(probabilities * probabilities))
    excess = (mode_count * concentration - 1.0) / (mode_count - 1.0)
    return float(np.sqrt(np.clip(excess, 0.0, 1.0)))


def _spatial_band_mask(lattice_size: int, band: tuple[float, float]) -> np.ndarray:
    lower, upper = map(float, band)
    if not 0.0 < lower < upper <= 1.0:
        raise ValueError(f"band must satisfy 0 < lower < upper <= 1 in units of pi, got {band}")
    frequencies_over_pi = 2.0 * np.fft.rfftfreq(int(lattice_size))[1:]
    mask = (frequencies_over_pi >= lower) & (frequencies_over_pi <= upper)
    if not np.any(mask):
        raise ValueError(f"band {band} contains no Fourier mode for L={lattice_size}")
    return mask


def selected_band_power(
    field: np.ndarray,
    band: tuple[float, float] = (0.25, 0.45),
) -> float:
    """Fraction of spatial power in a frozen selected-pattern band.

    Wavenumbers are expressed as fractions of pi.  The default band was fixed
    from the low-alpha discovery fields before full-lattice confirmation.
    """

    values = _as_field(field)
    probabilities = spatial_power_distribution(values)
    return float(probabilities[_spatial_band_mask(values.shape[1], band)].sum())


def selected_spatial_peak(
    field: np.ndarray,
    band: tuple[float, float] = (0.25, 0.45),
) -> float:
    """Largest normalised structure-factor mode in a frozen pattern band.

    Unlike integrated band mass, this quantity tends to zero as 1/L for a
    broadband field, while remaining finite for coherent spatial order.
    """

    values = _as_field(field)
    probabilities = spatial_power_distribution(values)
    return float(probabilities[_spatial_band_mask(values.shape[1], band)].max())


def temporal_spectral_entropy(field: np.ndarray) -> float:
    """Normalised entropy of the site-averaged non-DC temporal spectrum."""

    values = _as_field(field)
    centred = values - values.mean(axis=0, keepdims=True)
    power = np.mean(np.abs(np.fft.rfft(centred, axis=0)) ** 2, axis=1)[1:]
    total = float(power.sum())
    probabilities = np.zeros_like(power) if total <= 1e-15 else power / total
    return _normalised_entropy(probabilities)


def _spatial_slope_codes(field: np.ndarray, word_length: int) -> np.ndarray:
    values = _as_field(field)
    word_length = int(word_length)
    if not 2 <= word_length <= 8:
        raise ValueError(f"word_length must be in 2..8, got {word_length}")
    slopes = np.roll(values, -1, axis=1) > values
    codes = np.zeros(values.shape, dtype=np.uint16)
    for offset in range(word_length):
        codes |= np.roll(slopes, -offset, axis=1).astype(np.uint16) << offset
    return codes


def spatial_pattern_distribution(field: np.ndarray, word_length: int = 4) -> np.ndarray:
    """Distribution of circular binary spatial-slope words."""

    codes = _spatial_slope_codes(field, word_length)
    counts = np.bincount(codes.ravel(), minlength=2 ** int(word_length)).astype(np.float64)
    return counts / counts.sum()


def static_spatial_pattern_entropy(field: np.ndarray, word_length: int = 4) -> float:
    """Normalised Shannon entropy of spatial-slope patterns."""

    return _normalised_entropy(spatial_pattern_distribution(field, word_length))


def dynamical_spatial_pattern_entropy(field: np.ndarray, word_length: int = 4) -> float:
    """Normalised conditional entropy H(pattern[t+1] | pattern[t])."""

    codes = _spatial_slope_codes(field, word_length)
    alphabet_size = 2 ** int(word_length)
    pair_codes = codes[:-1].astype(np.int64) * alphabet_size + codes[1:]
    joint = np.bincount(pair_codes.ravel(), minlength=alphabet_size**2).astype(np.float64)
    joint = joint.reshape(alphabet_size, alphabet_size)
    joint /= joint.sum()
    current = joint.sum(axis=1, keepdims=True)
    positive = joint > 0
    conditional = -np.sum(joint[positive] * np.log((joint / np.maximum(current, 1e-300))[positive]))
    return float(conditional / np.log(alphabet_size))


def turbulent_fraction(field: np.ndarray, threshold: float) -> float:
    """Fraction of site-times whose period-two residual exceeds ``threshold``.

    This becomes a physical order-parameter candidate only after the activity
    rule and threshold sensitivity are established independently of SPI data.
    """

    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    return float(np.mean(period2_residual(field) > float(threshold)))


def spatial_correlation(field: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Equal-time ring correlation C(r), normalized so C(0)=1."""

    values = _as_field(field)
    lattice_size = values.shape[1]
    if max_lag is None:
        max_lag = min(64, lattice_size // 2)
    max_lag = int(max_lag)
    if not 1 <= max_lag <= lattice_size // 2:
        raise ValueError(
            f"max_lag must be in 1..{lattice_size // 2} for L={lattice_size}, got {max_lag}"
        )
    centred = values - values.mean()
    variance = float(np.mean(centred * centred))
    if variance <= 1e-15:
        return np.ones(max_lag + 1, dtype=np.float64)
    return np.array(
        [np.mean(centred * np.roll(centred, -lag, axis=1)) / variance for lag in range(max_lag + 1)],
        dtype=np.float64,
    )


def correlation_length_1e(correlation: np.ndarray) -> float:
    """First interpolated lag at which |C(r)| falls below exp(-1)."""

    corr = np.abs(np.asarray(correlation, dtype=np.float64))
    if corr.ndim != 1 or len(corr) < 2:
        raise ValueError("correlation must be a one-dimensional array of length >= 2")
    target = np.exp(-1.0)
    below = np.flatnonzero(corr[1:] <= target)
    if len(below) == 0:
        return float(len(corr) - 1)
    hi = int(below[0] + 1)
    lo = hi - 1
    y0, y1 = corr[lo], corr[hi]
    if abs(y1 - y0) < 1e-15:
        return float(hi)
    return float(lo + (target - y0) / (y1 - y0))


def largest_lyapunov_exponent(
    *,
    alpha: float,
    eps: float,
    lattice_size: int,
    seed: int,
    transients: int,
    steps: int,
) -> float:
    """Largest exponent from the exact CML tangent map (Benettin method)."""

    lattice_size = int(lattice_size)
    steps = int(steps)
    if lattice_size < 3 or steps <= 0:
        raise ValueError("lattice_size must be >=3 and steps must be positive")
    rng = np.random.default_rng(int(seed))
    state = rng.random(lattice_size)

    def advance(x: np.ndarray) -> np.ndarray:
        mapped = 1.0 - float(alpha) * x * x
        return (1.0 - eps) * mapped + (eps / 2.0) * (
            np.roll(mapped, 1) + np.roll(mapped, -1)
        )

    for _ in range(max(0, int(transients))):
        state = advance(state)
    tangent = rng.standard_normal(lattice_size)
    tangent /= np.linalg.norm(tangent)
    accumulated = 0.0
    for _ in range(steps):
        local = (-2.0 * alpha * state) * tangent
        propagated = (1.0 - eps) * local + (eps / 2.0) * (
            np.roll(local, 1) + np.roll(local, -1)
        )
        norm = float(np.linalg.norm(propagated))
        if norm <= 1e-300:
            return float("-inf")
        accumulated += np.log(norm)
        tangent = propagated / norm
        state = advance(state)
    return float(accumulated / steps)


def summarize_field(
    field: np.ndarray,
    *,
    activity_thresholds: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2),
    max_spatial_lag: int | None = None,
    selected_spatial_band: tuple[float, float] = (0.25, 0.45),
    pattern_word_length: int = 4,
) -> dict[str, float | list[float] | dict[str, float] | None]:
    """Compute a compact, JSON-serialisable physical-diagnostic record."""

    values = _as_field(field)
    residual = period2_residual(values)
    correlation = spatial_correlation(values, max_lag=max_spatial_lag)
    flat0 = values[:-1].ravel()
    flat1 = values[1:].ravel()
    lag1 = float(np.corrcoef(flat0, flat1)[0, 1])
    neighbour = float(
        np.corrcoef(values.ravel(), np.roll(values, -1, axis=1).ravel())[0, 1]
    )
    fractions = {
        f"{float(threshold):g}": float(np.mean(residual > float(threshold)))
        for threshold in activity_thresholds
    }
    spatial_power = spatial_power_distribution(values)
    try:
        pattern_peak = selected_spatial_peak(values, selected_spatial_band)
        pattern_band_power = selected_band_power(values, selected_spatial_band)
    except ValueError:
        # Very small test fields may have no discrete mode inside the frozen
        # physical band. Real scout lattices are much larger.
        pattern_peak = None
        pattern_band_power = None
    return {
        "field_mean": float(values.mean()),
        "field_std": float(values.std()),
        "period2_activity": float(np.sqrt(np.mean(residual * residual))),
        "period2_mean_abs": float(residual.mean()),
        "turbulent_fraction": fractions,
        "selected_spatial_band": list(map(float, selected_spatial_band)),
        "selected_spatial_peak": pattern_peak,
        "selected_band_power": pattern_band_power,
        "spatial_spectral_entropy": _normalised_entropy(spatial_power),
        "spatial_spectral_concentration": float(np.sum(spatial_power * spatial_power)),
        "spatial_order_magnitude": spatial_order_magnitude(values),
        "spatial_power_distribution": spatial_power.tolist(),
        "spatial_pattern_word_length": int(pattern_word_length),
        "static_spatial_pattern_entropy": static_spatial_pattern_entropy(
            values, pattern_word_length
        ),
        "dynamical_spatial_pattern_entropy": dynamical_spatial_pattern_entropy(
            values, pattern_word_length
        ),
        "temporal_spectral_entropy": temporal_spectral_entropy(values),
        "temporal_lag1_correlation": lag1,
        "spatial_neighbour_correlation": neighbour,
        "spatial_correlation": correlation.tolist(),
        "spatial_correlation_length_1e": correlation_length_1e(correlation),
    }
