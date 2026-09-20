import numpy as np
import pytest
from scipy.integrate import solve_ivp

from scripts.hindmarsh_rose_spike_adding import (
    rhs, integrate, burst_statistics, diagnostics, pilot_cases, simulate,
)


def test_equation_and_independent_solver():
    state = np.array([-.8, -4., 2.5])
    b = 2.68
    np.testing.assert_allclose(rhs(state, b), [-4.-(-.8)**3+b*.64-2.5+2.4,
                                             1.-5.*.64+4., .01*(4.*.8-2.5)])
    def independent(t, s):
        x, y, z = s
        return [y-x**3+b*x*x-z+2.4, 1-5*x*x-y, .01*(4*(x+1.6)-z)]
    reference = solve_ivp(independent, (0, 20), state, method="DOP853", rtol=1e-12, atol=1e-13)
    actual, _ = integrate(state, b, .001, 20000)
    np.testing.assert_allclose(actual, reference.y[:, -1], atol=2e-8, rtol=2e-8)


def synthetic_trace():
    dt = .01
    t = np.arange(0, 100, dt)
    phase = t % 20
    # Exactly three large maxima plus one small maximum; no flat regions or
    # Gaussian underflow plateaus introduce accidental floating-point maxima.
    x = np.interp(phase, [0, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 20],
                  [-2, -1.2, -.7, 1.5, -.7, 1.5, -.7, 1.5, -.7, -.65, -.75, -1.2, -2.2, -2])
    return x, dt


def test_ground_truth_counts_and_partial_bursts():
    x, dt = synthetic_trace()
    stats, _, _ = burst_statistics(x, dt)
    assert stats["Q"] == 3
    assert stats["complete_bursts"] == 4
    assert set(stats["all_local_maxima_counts"]) == {4}
    assert stats["mean_duty_cycle"] == pytest.approx(.45, abs=.002)
    assert diagnostics(x, dt)[0]["threshold_agreement"]
    assert burst_statistics(x[500:-500], dt)[0]["Q"] == 3


def test_constant_or_short_trace_not_claimed_as_bursting():
    stats, _, _ = burst_statistics(np.ones(100), .1)
    assert not stats["valid_burst_segmentation"]
    assert stats["Q"] is None
    with pytest.raises(ValueError):
        burst_statistics(np.array([1., np.nan, 2.]), .1)


def test_manifest_and_short_simulation():
    cases = pilot_cases(seeds=8)
    assert len(cases) == 255
    assert sum(c["arm"] == "primary" for c in cases) == 168
    assert {c["preparation"] for c in cases} == {"independent", "up", "down"}
    arrays, meta = simulate(2.68, 9, burn=20., reference=200., observation_samples=400)
    assert arrays["X"].shape == (3, 400)
    assert np.isfinite(arrays["X"]).all()
    assert meta["M"] == meta["N_state"] == 3
    assert meta["N_neurons"] == 1
    assert len(meta["input_sha256"]) == 64


def test_endpoint_physics_and_step_refinement():
    results = []
    for b in (2.68, 2.688):
        summaries = []
        for dt in (.01, .005):
            _, meta = simulate(b, 260915201, dt=dt, burn=2000., reference=1500., observation_samples=1000)
            summaries.append(meta["reference_summary"])
        assert summaries[0]["Q"] == summaries[1]["Q"]
        assert all(s["valid_burst_segmentation"] and s["threshold_agreement"] for s in summaries)
        assert all(s["Q_half_difference"] == 0 for s in summaries)
        results.append(summaries[0]["Q"])
    assert results == [6., 5.]


def test_preparation_selects_distinct_stable_branches():
    results = []
    for prep in ("up", "down"):
        _, meta = simulate(2.6819, 260915201, preparation=prep,
                           burn=2000., reference=1500., observation_samples=1000)
        stats = meta["reference_summary"]
        assert stats["threshold_agreement"] and stats["Q_half_difference"] == 0
        results.append(stats["Q"])
    assert results == [6., 5.]
