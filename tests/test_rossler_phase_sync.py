import numpy as np
import pytest
from scipy.integrate import solve_ivp

from scripts.rossler_phase_sync import (advance, combine_stats, initial_state,
    measure, pilot_cases, rhs, rk4_step, simulate, summarize)


def test_rhs_matches_published_equations_and_exchange_symmetry():
    state = np.array([2., -1., .3, -3., 4., .7])
    expected = [1.015 - .3 + .03 * -5, 1.015 * 2 - .15,
                .2 + .3 * -8, -.985 * 4 - .7 + .03 * 5,
                -.985 * 3 + .15 * 4, .2 + .7 * -13]
    np.testing.assert_allclose(rhs(state, .03), expected)
    swapped = np.r_[state[3:], state[:3]]
    derivative = rhs(state, .03)
    np.testing.assert_allclose(rhs(swapped, .03, -.015), np.r_[derivative[3:], derivative[:3]])


def test_rk4_matches_independent_solver_and_refines():
    state = initial_state(2, .03)
    truth = solve_ivp(lambda _, x: rhs(x, .03), [0., 1.], state,
                     method="DOP853", atol=1e-13, rtol=1e-13).y[:, -1]
    coarse = advance(state, .03, .02, 50)
    fine = advance(state, .03, .01, 100)
    assert np.linalg.norm(fine - truth) < np.linalg.norm(coarse - truth) / 10
    np.testing.assert_allclose(fine, truth, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(state, initial_state(2, .03))


def test_full_state_observation_exact_time_and_future_separation():
    arrays, metadata = simulate(.03, 7, burn=1., reference=2., observation_samples=10)
    state = advance(arrays["initial_state"], .03, .01, 100)
    for t in range(10):
        state = advance(state, .03, .01, 20)
        np.testing.assert_array_equal(arrays["X"][:, t], state)
    np.testing.assert_array_equal(arrays["reference_start_state"], state)
    assert arrays["X"].shape == (6, 10)
    assert metadata["observation_interval"] == [1.2, 3.]
    assert metadata["reference_interval"] == [3., 5.]
    assert metadata["N_state"] == metadata["M"] == 6
    assert len(metadata["reference_block_summaries"]) == 10


def test_streamed_phase_statistics_match_unsplit_reference():
    state = advance(initial_state(1, .02), .02, .01, 1000)
    final, _, complete = measure(state, .02, .01, 1000)
    stats = []
    for _ in range(10):
        state, _, chunk = measure(state, .02, .01, 100)
        stats.append(chunk)
    merged = combine_stats(np.array(stats))
    np.testing.assert_array_equal(final, state)
    # Slip anchors reset across blocks; other sufficient statistics combine exactly.
    np.testing.assert_allclose(merged[:12], complete[:12], atol=1e-12, rtol=0)
    np.testing.assert_allclose(merged[14:], complete[14:], atol=1e-9, rtol=0)
    assert summarize(merged, 1000, .01)["Q"] >= 0


def test_reproducibility_grid_and_validation():
    kwargs = dict(burn=0., reference=1., observation_samples=5)
    a, ma = simulate(.03, 12, **kwargs)
    b, mb = simulate(.03, 12, **kwargs)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert ma["input_sha256"] == mb["input_sha256"]
    cases = pilot_cases()
    assert len(cases) == 171
    assert len({c["coupling"] for c in cases if c["arm"] == "primary"}) == 21
    for case in cases[-3:]:
        assert any(c["coupling"] == case["coupling"] and c["seed"] == case["seed"]
                   for c in cases[:-3])
    with pytest.raises(ValueError):
        simulate(.03, 1, reference=.11)
    with pytest.raises(ValueError):
        simulate(.03, 1, dt=.03)
