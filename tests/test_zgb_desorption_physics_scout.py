import numpy as np

from scripts.zgb_desorption_physics_scout import _simulate_zgb_k


def test_zgb_k_seed_is_reproducible_and_coverages_are_physical() -> None:
    first = _simulate_zgb_k(10, 0.53, 0.02, 20, 30, 41, False)
    second = _simulate_zgb_k(10, 0.53, 0.02, 20, 30, 41, False)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
        assert np.all(np.isfinite(left))
        assert np.all((left >= 0.0) & (left <= 1.0))


def test_zgb_k_limiting_adsorption_and_desorption_cases() -> None:
    co, oxygen, reactions = _simulate_zgb_k(8, 1.0, 0.0, 100, 5, 51, False)
    np.testing.assert_array_equal(co, 1.0)
    np.testing.assert_array_equal(oxygen, 0.0)
    np.testing.assert_array_equal(reactions, 0.0)

    co, oxygen, reactions = _simulate_zgb_k(8, 0.5, 1.0, 100, 5, 61, True)
    np.testing.assert_array_equal(co, 0.0)
    np.testing.assert_array_equal(oxygen, 0.0)
    np.testing.assert_array_equal(reactions, 0.0)
