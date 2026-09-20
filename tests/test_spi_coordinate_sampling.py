import numpy as np
from scripts.scout_spi_coordinate_sampling import coordinate_statistics


def test_bias_variance_rmse_identity_and_exact_coordinate():
    values = np.array([[.1, .5], [.2, .5], [.3, .5]])
    stats = coordinate_statistics(values, np.array([.15, .5]))
    np.testing.assert_allclose(stats['rmse']**2, stats['bias']**2 + stats['sd']**2)
    assert stats['rmse'][1] == 0
    np.testing.assert_array_equal(stats['valid_fraction'], [1, 1])


def test_conditional_statistics_keep_missingness_visible():
    stats = coordinate_statistics(np.array([[.1, np.nan], [np.nan, np.nan]]), np.array([0, .3]))
    np.testing.assert_allclose(stats['valid_fraction'], [.5, 0])
    assert stats['rmse'][0] == .1
    assert np.isnan(stats['rmse'][1])
