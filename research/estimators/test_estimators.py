"""Fast checks of the estimators' defining properties (the full construction validation is validate.py)."""
import numpy as np

from research.estimators.estimators import estimate
from research.estimators.systems import make

MAIN = ('nonlinearity', 'lag', 'variability', 'direction')


def test_monotone_distortion_leaves_main_axes_unchanged():
    x, y = make('linearity', .5, 4000, np.random.default_rng(0))
    a = estimate(x, y, floor=False)
    b = estimate(x, np.sinh(3 * y) / 3, floor=False)
    assert all(np.isclose(a[k], b[k], equal_nan=True) for k in MAIN)


def test_lag_share_is_sin_squared_phase():
    for theta in (0., .5, 1.):
        x, y = make('lag', theta, 2 ** 14, np.random.default_rng(1), strength=.3)
        assert abs(estimate(x, y, floor=False)['lag'] - theta) < .03


def test_variability_reads_on_off_and_phase_drift_and_not_fixed_phase():
    for system, theta, want in (('intermittency', .6, .6), ('lag_variability', .6, .6), ('lag', .6, 0.)):
        x, y = make(system, theta, 2 ** 15, np.random.default_rng(2), strength=.4 if system == 'intermittency' else None)
        assert abs(estimate(x, y, floor=False)['variability'] - want) < .05


def test_whitened_channel_has_no_mean_when_end_points_differ():
    """circular() leaves a mean when the end points differ; whitening must not amplify it (it swamped the
    cross-correlation that finds the fixed delay, giving a wrong delay and a spurious variability)."""
    from research.estimators.estimators import circular, whiten
    x, _ = make('lag', 0., 4000, np.random.default_rng(5))
    x[-1] += 3.
    assert abs(whiten(circular(x)).mean()) < 1e-9


def test_independent_pair_is_undefined():
    x, y = make('independent', 0., 2000, np.random.default_rng(3))
    e = estimate(x, y, np.random.default_rng(4))
    assert not any(e[f'{k}_defined'] for k in MAIN)


def test_ksg_matches_the_gaussian_closed_form():
    from research.estimators.estimators import _ksg
    rng = np.random.default_rng(6)
    x = rng.standard_normal(4000)
    y = .8 * x + .6 * rng.standard_normal(4000)
    assert abs(_ksg(x, y) - -.5 * np.log(1 - .8 ** 2)) < .05


def test_nonlinearity_mi_endpoints():
    from research.estimators.estimators import nonlinearity_mi
    for theta, want in ((0., 0.), (1., 1.)):
        x, y = make('linearity', theta, 2000, np.random.default_rng(7))
        assert abs(nonlinearity_mi(x, y, np.random.default_rng(8))['nonlinearity_mi'] - want) < .1


def test_nonlinearity_mi_is_zero_for_identical_and_undefined_for_independent_channels():
    from research.estimators.estimators import nonlinearity_mi
    x, y = make('independent', 0., 2000, np.random.default_rng(3))
    assert nonlinearity_mi(x, x)['nonlinearity_mi'] == 0
    e = nonlinearity_mi(x, y, np.random.default_rng(4))
    assert not e['nonlinearity_mi_defined'] and np.isnan(e['nonlinearity_mi'])
