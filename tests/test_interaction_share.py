import numpy as np

from scripts.check_interaction_share_feasibility import drift_and_jacobian, interaction_share, ring


def test_jacobian_matches_intervention_derivative_and_is_contractive():
    w = ring(8)
    x = np.linspace(-2, 2, 8)
    for family in ["linear", "tanh"]:
        _, jacobian = drift_and_jacobian(x, .3, .5, w, family)
        numerical = np.column_stack([
            (drift_and_jacobian(x + e, .3, .5, w, family)[0] -
             drift_and_jacobian(x - e, .3, .5, w, family)[0]) / 2e-6
            for e in 1e-6 * np.eye(8)])
        np.testing.assert_allclose(jacobian, numerical, atol=1e-9)
        assert np.linalg.norm(jacobian, ord=2) <= .8 + 1e-12


def test_linear_interaction_share_has_known_value_and_is_relabeling_invariant():
    _, jacobian = drift_and_jacobian(np.zeros(8), .3, .5, ring(8), "linear")
    q = interaction_share(jacobian[None])
    np.testing.assert_allclose(q, (.5**2 / 2) / (.3**2 + .5**2 / 2))
    p = [3, 6, 1, 7, 0, 5, 2, 4]
    np.testing.assert_allclose(q, interaction_share(jacobian[np.ix_(p, p)][None]))
