"""Numerical check of the group account used in the sensitivity analysis."""
import numpy as np
from numpy.testing import assert_allclose

from research.empirical.hardware_pilot.si_ids_sensitivity import _partition


def test_group_mixture_can_reverse_negative_within_group_agreement():
    a = np.array([0., 1., 2., 10., 11., 12.])
    b = np.array([2., 1., 0., 12., 11., 10.])
    group = np.array([True, True, True, False, False, False])
    result = _partition(a, b, group)
    assert result['within_coherent'] < 0 and result['within_other'] < 0
    assert result['between'] > 0
    assert_allclose(sum(result.values()), np.corrcoef(a, b)[0, 1], atol=1e-14)
