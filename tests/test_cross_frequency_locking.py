import numpy as np
import pandas as pd
from scripts import cross_frequency_locking as c


def test_mean_field_velocity_matches_published_pairwise_sums():
    rng = np.random.default_rng(3)
    a, b, d = rng.uniform(0, 2*np.pi, (3, 5))
    wa, wb, wc = rng.normal(size=(3, 5))
    gamma, eps = .17, .4
    da, db, dc, _ = c.velocity(a, b, d, gamma, wa, wb, wc, eps)
    # Komarov & Pikovsky Eq. (10) with zero phase shifts.
    expected_a = wa + eps*np.sin(a[None] - a[:, None]).mean(1) + gamma*np.sin(b[None] - 2*a[:, None]).mean(1)
    expected_b = wb + eps*np.sin(b[None] - b[:, None]).mean(1) + gamma*np.sin(2*a[None] - b[:, None]).mean(1)
    expected_c = wc + eps*np.sin(d[None] - d[:, None]).mean(1)
    np.testing.assert_allclose(da, expected_a, atol=1e-13)
    np.testing.assert_allclose(db, expected_b, atol=1e-13)
    np.testing.assert_allclose(dc, expected_c, atol=1e-13)


def test_drifting_and_locked_sides_of_the_reduced_boundary():
    kwargs = dict(burn=200., reference=400., samples=200)
    _, drift = c.simulate(0., 0, **kwargs)
    _, locked = c.simulate(.2, 0, **kwargs)
    # Uncoupled collective phases slip at the mismatch; well above delta/(X2+2Y) they do not slip.
    np.testing.assert_allclose(drift['slip'], c.DELTA, atol=.03)
    assert drift['Q_lock'] < .2 and locked['Q_lock'] > .95 and locked['slip'] < .01
    assert c.DELTA/(locked['X2'] + 2*locked['Y']) < .12
    for truth in (drift, locked):
        assert min(truth['X1'], truth['Y'], truth['RC']) > .95


def test_records_are_reproducible_independent_and_nondegenerate():
    x, truth = c.record((.1, 1))
    again, same = c.record((.1, 1))
    np.testing.assert_array_equal(x, again)
    assert truth == same and x.shape == (24, 1000) and np.isfinite(x).all()
    np.testing.assert_allclose(x.mean(1), 0, atol=1e-12)
    np.testing.assert_allclose(x.std(1), 1, atol=1e-12)
    assert truth['max_r'] < 1 - 1e-6 and truth['abs_r_AB'] < .05 and truth['abs_r_within'] > .9
    assert not np.array_equal(x, c.record((.1, 2))[0]) and not np.array_equal(x, c.record((.11, 1))[0])


def test_index_partitions_cover_every_record_once():
    total = len(c.GAMMAS)*c.SEEDS
    parts = c.partitions(total)
    assert (len(parts['smoke']), len(parts['node'])) == (2, 48)
    assert sorted(sum(parts.values(), [])) == list(range(1, total + 1))


def test_step_scores_separate_a_step_from_a_drift():
    rng = np.random.default_rng(0)
    control = np.repeat(c.GAMMAS, 8)
    target = (control > .1).astype(float)
    frame = pd.DataFrame(dict(control=control, Q_lock=target,
                              step=target + .02*rng.normal(size=len(control)),
                              drift=control + .002*rng.normal(size=len(control))))
    step, drift = c.step_scores(frame, 'step'), c.step_scores(frame, 'drift')
    assert step['step_share'] > .9 and step['post_locking_drift'] < .1 and step['steepest_interval'] == '0.100-0.110'
    assert abs(drift['step_share'] - .2) < .05 and drift['post_locking_drift'] > .3


def test_sensor_noise_arms_are_reproducible_and_leave_truth_untouched():
    from scripts import cross_frequency_locking_snr as s
    fixed, truth = s.record(('fixed-noise', .1, 100))
    again, same = s.record(('fixed-noise', .1, 100))
    np.testing.assert_array_equal(fixed, again)
    assert truth == same and truth['eta'] == .5 and fixed.shape == (24, 1000)
    np.testing.assert_allclose(fixed.std(1), 1, atol=1e-12)
    # Noise SD .5 attenuates a within-community correlation near .98 by 1/(1+.25).
    np.testing.assert_allclose(truth['abs_r_within'], .98/1.25, atol=.02)
    assert truth['Q_lock'] == c.simulate(.1, 100)[1]['Q_lock'] and truth['condition'] < 50
    etas = [s.record(('random-noise', .1, 200 + k))[1]['eta'] for k in range(4)]
    assert min(etas) >= .3 and max(etas) <= .9 and len(set(etas)) == 4


def test_confirmation_arms_apply_one_nuisance_each():
    from scripts import cross_frequency_locking_confirm as k
    out={arm:k.record((arm,.1,k.RUNS['m48']['first'][arm])) for arm in k.ARMS}
    for arm,(x,truth) in out.items():
        assert x.shape==(48,1000) and np.isfinite(x).all() and truth['nuisance']==truth[dict(zip(k.ARMS,('eta','common','eps')))[arm]]
    noise,common,coupling=(out[arm][1] for arm in k.ARMS)
    assert .3<=noise['eta']<=.9 and (noise['common'],noise['eps'])==(0.,c.EPS)
    assert 0<=common['common']<=.8 and (common['eta'],common['eps'])==(.5,c.EPS)
    assert .15<=coupling['eps']<=.6 and (coupling['eta'],coupling['common'])==(.5,0.)
    # A shared signal of SD s gives unrelated channels a correlation s^2/(1+eta^2+s^2).
    s=common['common'];np.testing.assert_allclose(common['abs_r_AB'],s*s/(1.25+s*s),atol=.03)
    np.testing.assert_array_equal(out['common-mode'][0],k.record(('common-mode',.1,400))[0])
