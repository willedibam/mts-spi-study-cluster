import numpy as np
import pytest

from scripts.tasep_phase_boundary import (
    exact_stationary, small_generator_stationary, simulate, cases_from_config,
)


@pytest.mark.parametrize('N', [1, 2, 3, 5, 7])
@pytest.mark.parametrize('alpha,beta', [(.2,.2), (.15,.2), (.25,.2), (.7,.8)])
def test_dehp_matches_independent_generator(N, alpha, beta):
    states, pi, generator = small_generator_stationary(N, alpha, beta)
    np.testing.assert_allclose(pi@generator, 0, atol=2e-15)
    assert pi.min() > 0
    exact = exact_stationary(N, alpha, beta)
    assert exact['density'] == pytest.approx(pi@states.mean(axis=1), abs=2e-13)
    assert exact['current'] == pytest.approx(beta*(pi@states[:,-1]), abs=2e-13)


@pytest.mark.parametrize('N', [1, 6, 32, 64])
def test_exact_symmetry_and_product_line(N):
    assert exact_stationary(N,.2,.2)['density'] == pytest.approx(.5, abs=5e-14)
    assert exact_stationary(N,.3,.7)['density'] == pytest.approx(.3, abs=5e-14)
    assert exact_stationary(N,.3,.7)['current'] == pytest.approx(.21, abs=5e-14)
    assert exact_stationary(N,.15,.2)['density'] == pytest.approx(1-exact_stationary(N,.2,.15)['density'], abs=5e-14)


def test_uniform_time_sampling_and_stationary_density():
    # N=1 exactly has occupation alpha/(alpha+beta), unlike event-index
    # sampling, whose alternating occupied/empty sequence averages one half.
    arrays, meta = simulate(dict(N=1,alpha=.2,beta=.8,seed=73),
        dict(burn=1000,observation_steps=100000,reference_time=100000,
             reference_blocks=20,reference_trace_dt=10))
    assert abs(arrays['observed'].mean()-.2) < .01
    assert abs(meta['Q_reference']-.2) < .01
    assert meta['constant_channels'] == 0


def test_reproducibility_contract_and_multisite_mean():
    config = dict(burn=1000, observation_steps=2000,reference_time=50000,reference_blocks=20)
    case = dict(N=8,alpha=.2,beta=.2,seed=130)
    a, meta = simulate(case, config); b, _ = simulate(case, config)
    for key in a: np.testing.assert_array_equal(a[key],b[key])
    assert a['observed'].shape == (8,2000)
    assert set(np.unique(a['observed'])) == {0,1}
    assert meta['reference_start'] == 3000
    assert meta['Q_reference'] == pytest.approx(np.mean(meta['Q_blocks']))
    assert abs(meta['Q_reference']-.5) < .03
    assert meta['estimated_tau_int'] > 0
    assert meta['constant_channels'] == 0


def test_gillespie_full_configuration_distribution_matches_ctmc():
    states, stationary, _ = small_generator_stationary(3,.15,.2)
    arrays, _ = simulate(dict(N=3,alpha=.15,beta=.2,seed=513),
        dict(burn=1000,observation_steps=200000,reference_time=1000,
             reference_blocks=10))
    configurations = (arrays['observed'].T*(1 << np.arange(3))).sum(axis=1)
    empirical = np.bincount(configurations, minlength=len(states))/len(configurations)
    np.testing.assert_allclose(empirical, stationary, atol=.012, rtol=0)


def test_case_grid_and_invalid_inputs():
    assert len(cases_from_config({})) == 336
    assert len(cases_from_config(dict(grid={},cases=[dict(N=32,alpha=.2,seed=1)]))) == 1
    with pytest.raises(ValueError): exact_stationary(32,0,.2)
    with pytest.raises(ValueError): simulate(dict(N=32,alpha=.2,seed=1,initial='other'))
