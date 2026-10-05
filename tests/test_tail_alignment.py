import numpy as np
from scipy.stats import multivariate_t
from scripts.scout_tail_alignment import entropy,population,recording


def test_population_means_match_but_joint_structure_and_tail_quantity_change():
    rows=[population(j) for j in range(5)]
    for key in ['mean_r','mean_tau','mean_MI']:
        np.testing.assert_allclose([r[key] for r in rows],rows[0][key],rtol=0,atol=1e-14)
    assert np.all(np.diff([r['Q_tail'] for r in rows])>0)
    assert rows[-1]['z_r_MI']-rows[0]['z_r_MI']>.15


def test_entropy_against_scipy_and_recording_replay():
    for nu in [3.,30.]:
        for d in [1,2]:
            np.testing.assert_allclose(entropy(d,nu),multivariate_t(shape=np.eye(d),df=nu).entropy(),atol=1e-12)
    x=recording(2,8)
    assert x.shape==(16,1000) and np.isfinite(x).all()
    np.testing.assert_array_equal(x,recording(2,8))
    np.testing.assert_allclose(x.std(1),1,atol=1e-12)
