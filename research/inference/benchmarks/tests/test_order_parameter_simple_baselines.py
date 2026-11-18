import numpy as np
from research.inference.benchmarks.order_parameter_simple_baselines import input_statistics, attach
import pandas as pd


def test_signed_and_absolute_mpi_mean_are_distinct():
    t = np.linspace(0, 4 * np.pi, 1000)
    x = np.array([np.sin(t), -np.sin(t), np.sin(t)])
    result = input_statistics(x)
    np.testing.assert_allclose(result['mean_correlation'], -1 / 3)
    np.testing.assert_allclose(result['mean_abs_correlation'], 1)
    np.testing.assert_allclose(result['leading_correlation_fraction'], 1)


def test_zero_lag_mpi_equivalence_and_affine_invariance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(8, 500))
    result = input_statistics(x)
    mpi = np.corrcoef(x)
    np.fill_diagonal(mpi, np.nan)
    np.testing.assert_allclose(result['mean_correlation'], np.nanmean(mpi))
    np.testing.assert_allclose(result['mean_abs_correlation'], np.nanmean(abs(mpi)))
    changed = input_statistics(x * np.arange(1, 9)[:, None] + 17)
    for col in ['mean_correlation', 'mean_abs_correlation', 'temporal_spectral_entropy', 'leading_correlation_fraction']:
        np.testing.assert_allclose(result[col], changed[col])


def test_constant_channels_not_silently_omitted():
    x = np.array([np.ones(100), np.arange(100)])
    result = input_statistics(x)
    assert np.isnan(result['mean_correlation'])
    assert np.isnan(result['mean_abs_correlation'])


def test_existing_baselines_act_as_identity_check():
    frame = pd.DataFrame({'q': [1], 'mean_abs_correlation': [.5]})
    attached = attach(frame, [{'mean_abs_correlation': .5, 'mean_correlation': .2}])
    assert attached['q'].iloc[0] == 1
    try:
        attach(frame, [{'mean_abs_correlation': .7}])
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatch should fail')
