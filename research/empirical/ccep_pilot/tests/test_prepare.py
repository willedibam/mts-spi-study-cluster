import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.stats import pearsonr

from research.empirical.ccep_pilot.prepare import clean_windows, feature_vector, lag_matrix


def test_windows_are_disjoint_and_filter_margins_avoid_annotations():
    events = pd.DataFrame([dict(onset=60., duration=10., trial_type='artefact'),
                           dict(onset=600., duration=20., trial_type='seizure')])
    windows = clean_windows(events, 550, 32)
    assert len(windows) == 3
    assert all(b-a >= 32 for a,b in zip(windows, windows[1:]))
    for start in windows:
        assert start-4 >= 5 and start+36 <= 300
        assert start+36 <= 59 or start-4 >= 71


def test_lag_uses_physical_time_and_both_directions():
    x = np.random.default_rng(94).normal(size=(3, 4096))
    x[1] += np.roll(x[0], 16)
    expected = (pearsonr(x[0,16:],x[1,:-16]).statistic + pearsonr(x[1,16:],x[0,:-16]).statistic)/2
    assert_allclose(lag_matrix(x, 1/128, 2048)[0,1], expected, atol=1e-14)


def test_collapsed_profile_is_undefined_without_imputation():
    z = feature_vector({'constant':np.ones(8), 'ascending':np.arange(8.), 'descending':-np.arange(8.)})
    assert np.isnan(z[:2]).all()
    assert_allclose(z[2], -1.)
