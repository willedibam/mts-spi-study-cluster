import numpy as np
import pytest

from mtsspi.generators import (
    ar1_lag_attenuation,
    delay_alias_structure,
    generate_delay_aliased_mts,
    generate_series,
)


@pytest.mark.parametrize("max_lag_range", [0, 1, 7, 15])
def test_zero_lag_identity_and_positive_definite_for_every_lag_range(max_lag_range):
    target, _, loadings = delay_alias_structure(23, max_lag_range, 0.9, 0.6, 0.8, 3)
    floor = 1.0 - loadings.max() ** 2  # lambda_min(S0) bound, by the Schur argument
    for lag_range in np.linspace(0, max_lag_range, 4 * max_lag_range + 1):
        _, it = generate_delay_aliased_mts(
            23, 1, lag_range=lag_range, max_lag_range=max_lag_range,
            structure_seed=3, rng=0, return_internals=True,
        )
        assert np.array_equal(it.zero_lag_corr, target)
        recovered = it.intrinsic_corr * ar1_lag_attenuation(it.lags, 0.9)
        assert np.max(np.abs(recovered - target)) < 1e-12
        assert np.linalg.eigvalsh(it.intrinsic_corr)[0] >= floor - 1e-12
        assert it.lags.min() == 0
        assert it.lags.max() == int(np.floor(lag_range + 0.5))


def test_synchronous_and_maximal_ranges():
    _, sync = generate_delay_aliased_mts(12, 1, lag_range=0, rng=0, return_internals=True)
    _, full = generate_delay_aliased_mts(12, 1, lag_range=15, rng=0, return_internals=True)
    assert not sync.lags.any()
    assert np.array_equal(sync.intrinsic_corr, sync.zero_lag_corr)
    base = np.outer(full.loadings, full.loadings)
    np.fill_diagonal(base, 1.0)
    assert np.allclose(full.intrinsic_corr, base)
    with pytest.raises(ValueError):
        generate_delay_aliased_mts(12, 10, lag_range=16)


def test_finite_sample_zero_lag_and_lagged_correlation():
    M, T, a = 8, 40_000, 0.9
    data, it = generate_delay_aliased_mts(
        M, T, lag_range=6, max_lag_range=6, rng=11, return_internals=True
    )
    assert data.shape == (T, M)
    iu = np.triu_indices(M, 1)
    assert np.abs(np.corrcoef(data.T) - it.zero_lag_corr)[iu].mean() < 0.03
    j, k = 0, M - 1  # tau_k - tau_j = 6: peak cross-correlation at that shift
    d = it.lags[k] - it.lags[j]
    peak = np.corrcoef(data[: T - d, j], data[d:, k])[0, 1]
    assert abs(peak - it.intrinsic_corr[j, k]) < 0.05


def test_registry_and_structure_shared_across_rng():
    x = generate_series("delay_aliased_mts", seed=1, M=6, T=50, lag_range=3)
    y = generate_series("delay_aliased_mts", seed=1, M=6, T=50, lag_range=3)
    assert x.shape == (50, 6) and np.array_equal(x, y)
    _, a = generate_delay_aliased_mts(6, 5, rng=1, return_internals=True)
    _, b = generate_delay_aliased_mts(6, 5, rng=2, return_internals=True)
    assert np.array_equal(a.zero_lag_corr, b.zero_lag_corr)
