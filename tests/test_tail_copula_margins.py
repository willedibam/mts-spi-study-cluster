import numpy as np
from scipy.stats import kendalltau, kstest
from scripts import diagnose_tail_copula_margins as d
from scripts import scout_tail_alignment as original

T_T, T_G, G_T = d.ARMS


def test_construction_matches_original_parameters_and_tail_truth():
    for swaps in range(5):
        for mine, theirs in zip(d.parameters(swaps), original.parameters(swaps)):
            np.testing.assert_array_equal(mine, theirs)
        reference = original.population(swaps)
        for arm in (T_T, T_G):
            p = d.population(arm, swaps)
            np.testing.assert_allclose([p['Q_tail'], p['mean_tau'], p['mean_MI']],
                                       [reference['Q_tail'], reference['mean_tau'], reference['mean_MI']], rtol=1e-13)
        np.testing.assert_allclose(d.population(T_T, swaps)['z_Pearson_MI'], reference['z_r_MI'], rtol=1e-12)
        assert np.isnan(d.population(G_T, swaps)['Q_tail'])


def test_population_identities_that_apply():
    pop = {arm: [d.population(arm, j) for j in range(5)] for arm in d.ARMS}
    for arm in d.ARMS:
        for key in ('mean_tau', 'mean_MI'):
            assert np.ptp([p[key] for p in pop[arm]]) < 1e-14
    assert np.ptp([p['mean_r'] for p in pop[T_T]]) < 1e-14
    # Gaussian copula: Kendall and MI are both functions of rho alone, so their alignment cannot move.
    assert np.ptp([p['z_Kendall_MI'] for p in pop[G_T]]) < 1e-12
    for arm in (T_T, T_G):
        assert np.all(np.diff([p['z_Kendall_MI'] for p in pop[arm]]) > .01)
    # Changing margins or copula breaks exact Pearson matching: about 1% and 6% of mean r=.01 across the sweep.
    for arm, low, high in [(T_G, 5e-5, 2e-4), (G_T, 4e-4, 7e-4)]:
        assert low < np.ptp([p['mean_r'] for p in pop[arm]]) < high
    assert np.ptp([p['z_Pearson_MI'] for p in pop[G_T]]) > .03


def test_population_pearson_agrees_with_monte_carlo():
    rng = np.random.default_rng(7)
    n = 2_000_000
    for nu in (3., 30.):
        a, b = rng.normal(size=(2, n))
        partner = .2*a + np.sqrt(1 - .2**2)*b
        scale = np.sqrt(rng.chisquare(nu, n)/nu)
        gaussianized = np.corrcoef(d.to_gaussian(a/scale, nu), d.to_gaussian(partner/scale, nu))[0, 1]
        np.testing.assert_allclose(d.population_pearson(T_G, .2, nu), gaussianized, atol=3e-3)
    # t_3 margins have infinite fourth moment, so only the lighter tail has a usable Monte Carlo check.
    heavy = np.corrcoef(d.to_student(a, 30.), d.to_student(partner, 30.))[0, 1]
    np.testing.assert_allclose(d.population_pearson(G_T, .2, 30.), heavy, atol=3e-3)
    assert d.population_pearson(G_T, .2, 3.) < .2 and d.population_pearson(T_G, .2, 3.) < .2


def test_recordings_are_reproducible_independent_and_standardized():
    for arm in d.ARMS:
        x = d.recording(arm, 2, 5)
        np.testing.assert_array_equal(x, d.recording(arm, 2, 5))
        assert x.shape == (16, 1000) and np.isfinite(x).all()
        np.testing.assert_allclose(x.mean(1), 0, atol=1e-12)
        np.testing.assert_allclose(x.std(1), 1, atol=1e-12)
        for other in [(arm, 3, 5), (arm, 2, 6)]:
            assert not np.array_equal(x, d.recording(*other))
    assert not np.array_equal(d.recording(T_T, 2, 5), d.recording(T_G, 2, 5))


def test_analytic_margins_are_applied_before_standardization():
    nu = np.repeat(np.repeat([3., 30.], 4), 2)
    gaussian, _ = d.latent(T_G, 1, 3)
    heavy, _ = d.latent(G_T, 1, 3)
    for j in range(16):
        assert kstest(gaussian[j], 'norm').pvalue > 1e-3
        assert kstest(heavy[j], 't', args=(nu[j],)).pvalue > 1e-3
    np.testing.assert_allclose(d.to_student(d.to_gaussian(np.linspace(-40, 40, 81), 3.), 3.), np.linspace(-40, 40, 81), rtol=1e-9)
    # The PIT is strictly monotone per channel, so every rank statistic of the t-copula draw is unchanged.
    rng = np.random.default_rng(np.random.SeedSequence([d.SEED, d.ARMS.index(T_G), 1, 3]))
    rho, module_nu = d.parameters(1)
    noise = rng.normal(size=(8, 2, d.T))
    scale = np.sqrt(rng.chisquare(module_nu[:, None], size=(8, d.T))/module_nu[:, None])
    first = noise[:, 0]/scale
    second = (rho[:, None]*noise[:, 0] + np.sqrt(1 - rho[:, None]**2)*noise[:, 1])/scale
    for k in range(8):
        assert kendalltau(first[k], second[k]).statistic == kendalltau(gaussian[2*k], gaussian[2*k + 1]).statistic


def test_estimate_is_deterministic_and_complete():
    row, z = d.estimate((T_G, 4, 0))
    again, z_again = d.estimate((T_G, 4, 0))
    assert row == again
    np.testing.assert_array_equal(z, z_again)
    assert z.shape == (6,) and np.isfinite(z).all()
    assert all(np.isfinite(v) for v in row.values() if isinstance(v, float))
    assert row['role'] == 'fit' and d.estimate((T_G, 4, d.FIT_SEEDS))[0]['role'] == 'validation'
