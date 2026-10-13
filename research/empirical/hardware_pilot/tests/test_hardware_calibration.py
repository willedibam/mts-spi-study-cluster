"""Independent numerical checks for the selected-statistic interpretation."""
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import pearsonr

from research.empirical.hardware_pilot.calibrate_hardware_pilot import (
    FS, analytic_profiles, control_signal, edge_corr, lag_matrix,
)


def test_lag_profile_matches_scalar_bidirectional_pearson():
    rng = np.random.default_rng(94)
    x = rng.normal(size=(4, 101))
    x[1] += np.roll(x[0], 3)
    x[2] = -2 * x[0] + .2 * x[2]
    x += np.array([100., -7., .3, 0.])[:, None]
    for tau in (0, 1, 10):
        expected = np.empty((4, 4))
        for i in range(4):
            for j in range(4):
                expected[i, j] = pearsonr(x[i], x[j]).statistic if tau == 0 else .5 * (
                    pearsonr(x[i, tau:], x[j, :-tau]).statistic
                    + pearsonr(x[j, tau:], x[i, :-tau]).statistic)
        assert_allclose(lag_matrix(x, tau), expected, rtol=0, atol=1e-14)


def test_harmonic_identity_against_periodic_observations():
    rng = np.random.default_rng(815)
    phi, signs = rng.uniform(-1, 1, 28), rng.choice([-1., 1.], 28)
    x, _ = control_signal(phi, signs, amplitude=.31, third=.08, samples=FS)
    predictions = analytic_profiles(phi, signs, amplitude=.31, third=.08)
    for tau, predicted in predictions.items():
        circular = np.corrcoef(x, np.roll(x, tau, axis=1))[:28, 28:]
        assert_allclose(.5 * (circular + circular.T), predicted, rtol=0, atol=1e-12)


def test_profile_collapse_is_undefined():
    assert np.isnan(edge_corr(np.ones(6), np.arange(6.)))
    assert np.isnan(edge_corr(np.arange(6.), np.full(6, np.nan)))
    assert_allclose(edge_corr(np.arange(6.), -np.arange(6.)), -1., atol=1e-15)
