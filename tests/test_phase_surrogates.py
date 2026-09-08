import numpy as np
import pytest
from src.phase_surrogates import phase_surrogate, pooled_autospectrum, spectrum_checks


@pytest.mark.parametrize('length',[127,128])
def test_shared_and_independent_spectral_contracts(length):
    rng=np.random.default_rng(71)
    x=rng.normal(size=(length,4))+np.arange(4)[None]
    for shared in [False,True]:
        y=phase_surrogate(x,23,shared)
        spectrum_checks(x,y,shared)
        np.testing.assert_allclose(pooled_autospectrum(x),pooled_autospectrum(y),atol=1e-12)
        np.testing.assert_array_equal(y,phase_surrogate(x,23,shared))
        assert not np.allclose(x,y)


def test_shared_preserves_relative_channel_phases_independent_changes_them():
    rng=np.random.default_rng(9)
    x=np.repeat(rng.normal(size=(256,1)),4,axis=1)
    common=phase_surrogate(x,31,True)
    separate=phase_surrogate(x,31,False)
    np.testing.assert_allclose(common[:,0],common[:,3],atol=1e-12)
    assert not np.allclose(separate[:,0],separate[:,3])
    assert spectrum_checks(x,separate,False)['relative_cross_periodogram_error']>.1


def test_pooled_spectra_ignore_channel_permutation():
    x=np.random.default_rng(7).normal(size=(128,6))
    np.testing.assert_allclose(pooled_autospectrum(x),pooled_autospectrum(x[:,::-1]),atol=1e-12)
