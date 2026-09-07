import itertools
import numpy as np

from scripts.check_interaction_share_feasibility import drift_and_jacobian, ring
from scripts.check_interaction_share_references import parameters
from src.interaction_share_reference import fit_map, energy_share


def test_gain_nuisance_changes_dynamics_without_changing_linear_share():
    for value in [.1, .3, .5, .7, .9]:
        gains = []
        for seed in range(5):
            a, b, gain = parameters(value, "independent_gain", seed)
            np.testing.assert_allclose((b*b / 2) / (a*a + b*b / 2), value)
            np.testing.assert_allclose(a + b, gain)
            assert .25 <= gain <= .8
            gains.append(gain)
        assert np.ptp(gains) > .1


def test_reference_recovers_linear_and_nonlinear_drift():
    rng = np.random.default_rng(21)
    for family in ["linear", "tanh"]:
        x = rng.normal(size=4)
        data = []
        for _ in range(18000):
            data.append(x)
            x = drift_and_jacobian(x, .25, .5, ring(4), family)[0] + .5 * rng.normal(size=4)
        data = np.asarray(data)
        model = fit_map(data, nonlinear=(family == "tanh"), ridge_fraction=1e-5)
        expected = np.asarray([drift_and_jacobian(x, .25, .5, ring(4), family)[1] for x in data[:30]])
        np.testing.assert_allclose(model.jacobian(data[:30]), expected, atol=.07)
        own, cross = model.energies(data[:30])
        jac = model.jacobian(data[:30])
        np.testing.assert_allclose(own + cross, np.square(jac).sum(axis=(1, 2)).mean())


def test_fitted_jacobian_is_derivative_and_estimate_ignores_channel_names():
    rng = np.random.default_rng(13)
    raw = rng.normal(size=(200, 5))
    model = fit_map(raw, nonlinear=True)
    point = raw[0]
    numerical = np.column_stack([(model.predict(point + e) - model.predict(point - e)) / 2e-6 for e in 1e-6 * np.eye(5)])
    np.testing.assert_allclose(numerical, model.jacobian(point[None])[0], atol=1e-9)
    permutation = [3, 0, 4, 1, 2]
    shuffled = fit_map(raw[:, permutation], nonlinear=True)
    np.testing.assert_allclose(model.energies(raw), shuffled.energies(raw[:, permutation]), atol=1e-12)


def test_sampling_corrects_energies_not_expected_ratio():
    jac = .3 * np.eye(6) + .5 * ring(6)
    own, cross = [], []
    for indices in itertools.combinations(range(6), 3):
        block = jac[np.ix_(indices, indices)]
        d = np.trace(block**2); o = np.sum(block**2) - d
        own.append(d); cross.append(o)
    np.testing.assert_allclose(np.mean(own) / (3 / 6), np.trace(jac**2))
    np.testing.assert_allclose(np.mean(cross) / (3 * 2 / (6 * 5)), np.sum(jac**2) - np.trace(jac**2))
    expected_q = energy_share(np.trace(jac**2), np.sum(jac**2) - np.trace(jac**2), 6)
    assert abs(np.mean([energy_share(d, o, 3, 6) for d, o in zip(own, cross)]) - expected_q) > .01
