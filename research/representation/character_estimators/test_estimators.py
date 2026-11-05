"""Fast checks of the estimators' defining properties (the full construction validation is validate.py)."""
import numpy as np

from research.representation.character_estimators.estimators import estimate
from research.representation.character_estimators.systems import make

MAIN = ('nonlinearity', 'lag', 'lag_variability', 'intermittency', 'direction')


def test_monotone_distortion_leaves_main_axes_unchanged():
    x, y = make('linearity', .5, 4000, np.random.default_rng(0))
    a = estimate(x, y, floor=False)
    b = estimate(x, np.sinh(3 * y) / 3, floor=False)
    assert all(np.isclose(a[k], b[k], equal_nan=True) for k in MAIN)


def test_lag_share_is_sin_squared_phase():
    for theta in (0., .5, 1.):
        x, y = make('lag', theta, 2 ** 14, np.random.default_rng(1), strength=.3)
        assert abs(estimate(x, y, floor=False)['lag'] - theta) < .03


def test_on_off_coupling_gives_intermittency_one_minus_fraction():
    x, y = make('intermittency', .6, 2 ** 15, np.random.default_rng(2))
    e = estimate(x, y, floor=False)
    assert abs(e['intermittency'] - .6) < .05 and e['lag_variability'] < .05


def test_independent_pair_is_undefined():
    x, y = make('independent', 0., 2000, np.random.default_rng(3))
    e = estimate(x, y, np.random.default_rng(4))
    assert not any(e[f'{k}_defined'] for k in MAIN)
