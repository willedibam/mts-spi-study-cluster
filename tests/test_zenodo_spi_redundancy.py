import numpy as np
from scripts.scout_zenodo_spi_redundancy import screen


def test_redundancy_screen_separates_coverage_from_similarity():
    stats = screen(np.array([[1., .8, np.nan], [np.nan, 1., np.nan]]))
    np.testing.assert_allclose(stats['valid_fraction'], [.5, 1, 0])
    np.testing.assert_allclose(stats['fraction_valid_ge_095'][:2], [1, .5])
    assert np.isnan(stats['fraction_valid_ge_095'][2])
