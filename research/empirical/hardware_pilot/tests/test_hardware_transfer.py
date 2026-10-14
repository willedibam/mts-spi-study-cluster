import json

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.stats import pearsonr

from research.empirical.hardware_pilot.calibrate_hardware_pilot import BANDS, phase_properties
from research.empirical.hardware_pilot.transfer_hardware_pilot import (
    INITIAL_STEPS, band_properties, lag_profiles, matched_pairs, select_refinements,
)


def test_physical_lags_use_correct_samples_at_each_resolution():
    rng = np.random.default_rng(914)
    for fs in (500, 1000):
        x = rng.normal(size=(28, fs))
        profiles = lag_profiles(x, fs)
        for k, seconds in enumerate((.002, .020)):
            tau = round(seconds * fs)
            expected = .5 * (pearsonr(x[0, tau:], x[1, :-tau]).statistic
                             + pearsonr(x[1, tau:], x[0, :-tau]).statistic)
            assert_allclose(profiles[k, 0], expected, atol=1e-14)
    with pytest.raises(ValueError, match='physical lags'):
        lag_profiles(x, 333)


def test_band_definition_matches_er_exactly_at_frozen_rate():
    time = np.arange(5000) / 500
    phi = np.linspace(-.3, .3, 28)[:, None]
    x = np.cos(2 * np.pi * 116 * time + phi) + .2 * np.cos(2 * np.pi * 58 * time + phi / 2)
    for limits in BANDS.values():
        expected = phase_properties(x, limits)
        actual = band_properties(x, 500, limits)
        for key in expected:
            assert_allclose(actual[key], expected[key], rtol=0, atol=0)


def test_refinement_is_physical_only_and_frozen(tmp_path):
    frame = pd.DataFrame({'step': INITIAL_STEPS,
                          'frequency_range': [1 if step < 28 else .001 for step in INITIAL_STEPS]})
    frame.to_csv(tmp_path / 'physical-screen.csv', index=False)
    select_refinements(tmp_path)
    result = json.loads((tmp_path / 'selection.json').read_text())
    assert result['additional_steps'] == [22, 24, 25, 27]
    assert result['features_not_computed']
    assert len(result['all_steps']) == 16
    frame.loc[frame.step == 28, 'frequency_range'] = 1
    frame.to_csv(tmp_path / 'physical-screen.csv', index=False)
    with pytest.raises(ValueError, match='different frozen selection'):
        select_refinements(tmp_path)


def test_matched_pair_requires_stability_and_reports_every_eligible_pair():
    shared = dict(reconstruction_target_met=True, mean_pearson=.8)
    frame = pd.DataFrame([
        dict(shared, step=1, z=.1, half_1=.11, half_2=.09, delete_min=.08, delete_max=.12),
        dict(shared, step=2, z=.4, half_1=.39, half_2=.41, delete_min=.37, delete_max=.43),
        dict(shared, step=3, z=.5, half_1=.05, half_2=.8, delete_min=-.2, delete_max=.9),
    ])
    pairs = matched_pairs(frame)
    assert len(pairs) == 3
    assert pairs.working_contrast_target.tolist() == [True, False, False]
