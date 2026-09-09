import numpy as np

from scripts.analyze_vicsek_observation_scout import channel_diagnostics


def test_constant_channels_are_flagged_not_silently_correlated():
    x = np.column_stack([np.ones(20), np.arange(20), -np.arange(20)])
    result = channel_diagnostics(x)
    assert result['constant_fraction'] == 1 / 3
    np.testing.assert_allclose(result['mean_abs_correlation'], 1.)
    assert np.isnan(channel_diagnostics(np.ones((20, 3)))['mean_abs_correlation'])
