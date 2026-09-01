import numpy as np

from src.cml_order_parameter import (
    correlation_length_1e,
    dynamical_spatial_pattern_entropy,
    period2_activity,
    selected_band_power,
    selected_spatial_peak,
    spatial_correlation,
    spatial_order_magnitude,
    spatial_spectral_concentration,
    spatial_spectral_entropy,
    static_spatial_pattern_entropy,
    summarize_field,
    temporal_spectral_entropy,
    turbulent_fraction,
)
from src.generators.dynamical import generate_cml_logistic
from src.generators.order_parameter import generate_quadratic_cml_order_parameter


def test_period_two_field_has_zero_activity() -> None:
    field = np.tile(np.array([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]]), (10, 1))
    assert period2_activity(field) == 0.0
    assert turbulent_fraction(field, threshold=0.01) == 0.0
    assert summarize_field(field, max_spatial_lag=1)["period2_activity"] == 0.0


def test_spatial_correlation_is_normalized() -> None:
    field = np.random.default_rng(0).normal(size=(100, 16))
    correlation = spatial_correlation(field, max_lag=5)
    assert correlation.shape == (6,)
    assert correlation[0] == 1.0
    assert np.isfinite(correlation_length_1e(correlation))


def test_selected_spatial_mode_is_concentrated_and_predictable() -> None:
    sites = np.arange(60)
    pattern = np.cos(2 * np.pi * 10 * sites / 60)
    field = np.tile(pattern, (200, 1))
    assert selected_spatial_peak(field) > 0.999
    assert selected_band_power(field) > 0.999
    assert spatial_spectral_concentration(field) > 0.999
    assert spatial_order_magnitude(field) > 0.999
    assert spatial_spectral_entropy(field) < 1e-12
    assert dynamical_spatial_pattern_entropy(field) < 1e-12


def test_broadband_field_has_higher_spatial_and_temporal_entropy() -> None:
    rng = np.random.default_rng(2)
    broadband = rng.normal(size=(2000, 64))
    coherent = np.tile(np.cos(2 * np.pi * 10 * np.arange(64) / 64), (2000, 1))
    period_two = np.tile(np.array([-1.0, 1.0]), (1000, 64))
    assert spatial_spectral_entropy(broadband) > 0.98
    assert spatial_order_magnitude(broadband) < spatial_order_magnitude(coherent)
    assert static_spatial_pattern_entropy(broadband) > 0.9
    assert temporal_spectral_entropy(broadband) > 0.98
    assert temporal_spectral_entropy(period_two) < 1e-12


def test_explicit_lattice_size_returns_matching_crop() -> None:
    observed, full = generate_cml_logistic(
        M=7,
        T=13,
        alpha=1.75,
        eps=0.3,
        transients=11,
        sample_every=3,
        rng=np.random.default_rng(42),
        zscore=False,
        lattice_size=128,
        return_full_lattice=True,
    )
    assert observed.shape == (13, 7)
    assert full.shape == (13, 128)
    assert np.array_equal(observed, full[:, 60:67])


def test_distributed_cml_observations_are_nested() -> None:
    common = dict(
        T=20,
        alpha=1.8,
        eps=0.3,
        transients=50,
        lattice_size=64,
        observation_mode="distributed",
        return_full_lattice=True,
        return_observation_indices=True,
        zscore=False,
    )
    small, small_full, small_indices = generate_cml_logistic(
        M=8, rng=np.random.default_rng(67), **common
    )
    large, large_full, large_indices = generate_cml_logistic(
        M=16, rng=np.random.default_rng(67), **common
    )
    np.testing.assert_array_equal(small_full, large_full)
    np.testing.assert_array_equal(small_indices, large_indices[:8])
    np.testing.assert_array_equal(small, large[:, :8])


def test_streaming_generator_preserves_historical_sampling() -> None:
    observed = generate_cml_logistic(
        M=7,
        T=13,
        alpha=1.75,
        eps=0.3,
        transients=11,
        sample_every=3,
        rng=np.random.default_rng(42),
        zscore=False,
    )
    expected_start = np.array(
        [
            [0.59548644, 0.48696395, -0.28996537],
            [0.75353534, 0.81050646, 0.96867212],
        ]
    )
    np.testing.assert_allclose(observed[:2, :3], expected_start, rtol=0, atol=5e-9)


def test_quadratic_cml_future_truth_is_shared_across_nested_prefixes() -> None:
    common = dict(
        alpha=1.8,
        eps=0.3,
        lattice_size=32,
        transients=40,
        sample_every=2,
        truth_start_T=20,
        future_truth_T=60,
        observation_mode="distributed",
        return_internals=True,
        zscore=False,
    )
    short, short_info = generate_quadratic_cml_order_parameter(
        M=4,
        T=10,
        rng=np.random.default_rng(187),
        **common,
    )
    long, long_info = generate_quadratic_cml_order_parameter(
        M=8,
        T=20,
        rng=np.random.default_rng(187),
        **common,
    )
    np.testing.assert_array_equal(short, long[:10, :4])
    np.testing.assert_array_equal(
        short_info.observation_indices, long_info.observation_indices[:4]
    )
    for name in (
        "temporal_spectral_entropy",
        "dynamical_spatial_pattern_entropy",
        "selected_band_power",
        "period2_activity",
    ):
        assert short_info.truth_summary[name] == long_info.truth_summary[name]
