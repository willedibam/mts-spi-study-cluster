"""Edge-block bookkeeping for the locking character-axis audit."""
import numpy as np
import pandas as pd
import pytest

from research.inference.benchmarks import locking_character_axes as lca

FEATURES = lca.RESULTS/lca.RUN/'features.npz'


def test_masks_partition_the_ordered_pairs():
    m = lca.masks(8)
    counts = {k: int(v.sum()) for k, v in m.items()}
    assert counts == dict(AB=128, AA=56, BB=56, cross_C=256, without_AB=424)
    assert not (m['AB'] & m['cross_C']).any() and (m['AB'] | m['without_AB']).all()


def test_block_z_matches_corrcoef_and_marks_constant_spis():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(4, 552))
    V[3] = 1.0
    mask = lca.masks()['without_AB']
    C = lca._block_z(V, mask)
    np.testing.assert_allclose(C[:3, :3], np.corrcoef(V[:3, mask]), atol=1e-12)
    assert np.isnan(C[3]).all() and np.isnan(C[:, 3]).all()


@pytest.mark.skipif(not FEATURES.exists(), reason='locking feature bank not local')
def test_every_catalogue_spi_has_a_family():
    order = list(np.load(FEATURES)['spi_order'])
    assert len(order) == 289 and not [s for s in order if lca.family(s) == 'other']


def test_classify_uses_only_the_stated_thresholds():
    frame = pd.DataFrame(dict(abs_rho_Q=[.9, .9, .9, .7], abs_rho_eta=[.1, .1, .3, .1], change=[.5, .5, .5, .5],
                              AB_share=[.9, .1, .9, .9], family_a=['x']*4, family_b=['y']*4))
    out = lca.classify(frame)
    assert out.index.tolist() == [0, 1] and out.carrier.tolist() == ['A–B pairs', 'within communities']
