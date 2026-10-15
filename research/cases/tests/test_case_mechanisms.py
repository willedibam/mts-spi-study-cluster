"""Check the quantitative identities used in the mechanism viewers."""
import numpy as np
import pytest
from research.cases.case_mechanisms import correlation_decomposition, shifted_path, path_distance


def test_pair_type_covariance_separates_opposing_within_and_between_trends():
    x = np.array([0., 1., 10., 11.])
    y = np.array([1., 0., 11., 10.])
    within, between, total = correlation_decomposition(x, y, ['a', 'a', 'b', 'b'])
    np.testing.assert_allclose([within, between, total], [-1/101, 100/101, 99/101])
    one_group = correlation_decomposition(x, y, ['all']*4)
    np.testing.assert_allclose(one_group, [total, 0., total])


@pytest.mark.parametrize('shift', [-2, 0, 2])
def test_shift_paths_include_endpoints_and_have_the_archived_cost(shift):
    x = np.array([0., 2., -1., 4., 1.]); y = np.array([1., 0., 3., -2., 2.])
    path = shifted_path(len(x), shift)
    assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (4, 4)
    steps = np.diff(path, axis=0)
    assert np.all((steps >= 0) & (steps <= 1)) and np.all(steps.sum(axis=1) > 0)
    assert len(path) == len(x)+abs(shift)
    # Manually enumerated squared costs for these three paths, divided by T.
    expected = {-2: 40., 0: 58., 2: 36.}
    np.testing.assert_allclose(path_distance(x, y, path), np.sqrt(expected[shift]/5))
