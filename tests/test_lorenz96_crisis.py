import numpy as np
import pytest
from scipy.integrate import solve_ivp

from scripts.lorenz96_crisis import (advance, alternating_mode, label_update,
    pilot_cases, rhs, simulate, tangent_rhs)


def test_rhs_periodic_equivariance_and_energy_conservation():
    x = np.random.default_rng(1).normal(size=8)
    expected = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x - 6.5
    np.testing.assert_allclose(rhs(x, -6.5), expected)
    np.testing.assert_allclose(rhs(np.roll(x, 1), -6.5), np.roll(rhs(x, -6.5), 1))
    nonlinear = rhs(x, 0.) + x
    assert np.dot(x, nonlinear) == pytest.approx(0., abs=1e-14)
    assert alternating_mode(np.roll(x, 1)) == pytest.approx(-alternating_mode(x))


def test_tangent_matches_finite_difference_and_rk4_refines():
    rng = np.random.default_rng(3)
    x, v = rng.normal(size=(2, 8))
    np.testing.assert_allclose(tangent_rhs(np.r_[x, v], -6.5)[8:],
        (rhs(x + 1e-6*v, -6.5) - rhs(x - 1e-6*v, -6.5)) / 2e-6, atol=1e-8)
    state = np.r_[x, v / np.linalg.norm(v)]
    truth = solve_ivp(lambda _, y: rhs(y, -6.5), (0, .2), x,
                     method="DOP853", atol=1e-12, rtol=1e-12).y[:, -1]
    coarse = advance(state, -6.5, .02, 10, 10)[:8]
    fine = advance(state, -6.5, .01, 20, 20)[:8]
    assert np.linalg.norm(fine - truth) < np.linalg.norm(coarse - truth) / 10


def test_hysteresis_no_deadband_chatter():
    label = 0
    result = []
    for value in (.1, .3, .1, -.1, -.3, -.1, .3):
        label = label_update(value, label, .25)
        result.append(label)
    assert result == [0, 1, 1, 1, -1, -1, 1]


def test_full_observation_and_deterministic_streaming():
    kwargs = dict(burn=1., reference=10., observation_samples=20, anchor_duration=2.)
    arrays, meta = simulate(-6.4, 13, **kwargs)
    again, _ = simulate(-6.4, 13, **kwargs)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], again[key])
    assert arrays["X"].shape == (8, 20)
    assert arrays["anchor_X"].shape == (8, 20)
    np.testing.assert_allclose(arrays["anchor_mode"],
        (arrays["anchor_X"][::2].sum(axis=0) - arrays["anchor_X"][1::2].sum(axis=0)) / 8)
    assert meta["observation_interval"] == [1.1, 3.]
    assert meta["reference_interval"] == [3., 13.]
    for row in meta["residence_diagnostics"].values():
        assert sum(row["completed_count_by_state"]) == max(0, row["switch_count"] - 1)
        assert sum(row["occupancy_fraction_by_state"]) + row["unknown_duration"] / 10 == pytest.approx(1)
    assert "UNVALIDATED" in meta["classifier_status"]


def test_case_design_and_invalid_time():
    cases = pilot_cases()
    assert len(cases) == 171
    assert len({row["forcing"] for row in cases}) == 21
    assert sum(c["anchor_duration"] > 0 for c in cases) == 6
    with pytest.raises(ValueError):
        simulate(-6.4, 1, reference=11.)
    with pytest.raises(ValueError):
        simulate(-6.4, 1, dt=.03)
