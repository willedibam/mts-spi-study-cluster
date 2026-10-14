"""Mathematical replay and counting checks behind the pilot interpretation."""
import numpy as np
import pytest
from research.empirical.ccep_pilot.interpret import circular_lag, shared_count
from research.empirical.ccep_pilot.analyse import correlation_endpoint


@pytest.mark.parametrize('n',[63,64])
def test_cross_spectrum_replays_symmetric_circular_lag(n):
    x=np.random.default_rng(12).normal(size=(5,n))
    x=(x-x.mean(axis=1,keepdims=True))/x.std(axis=1,keepdims=True)
    actual,_,_=circular_lag(x,4)
    forward=x@np.roll(x,4,axis=1).T/n
    np.testing.assert_allclose(actual,(forward+forward.T)/2,atol=1e-12)


def test_band_removal_preserves_outside_frequencies():
    x=np.random.default_rng(2).normal(size=(4,4096))
    x=(x-x.mean(axis=1,keepdims=True))/x.std(axis=1,keepdims=True)
    _,fraction,y=circular_lag(x,4)
    f=np.fft.rfftfreq(4096,1/128);band=(f>8)&(f<24)
    assert 0<fraction<1
    np.testing.assert_allclose(np.fft.rfft(y)[:,band],0,atol=1e-11)
    np.testing.assert_allclose(np.fft.rfft(y)[:,~band],np.fft.rfft(x)[:,~band],atol=1e-11)


def test_shared_support_counts_contact_capacity_not_number_of_bins():
    assert shared_count([[3,4,0],[5,1,1],[1,8,2],[2,7,0]])==2
    assert shared_count([[8,0],[0,8]])==0


def test_correlation_filter_includes_rank_and_covariance_estimators():
    for name in ['spearmanr','kendalltau','cov_LedoitWolf','cov_GraphicalLasso','corr_pearson_tau-4']:
        assert correlation_endpoint(name)
    assert not correlation_endpoint('prec_ShrunkCovariance')
    assert not correlation_endpoint('pdist_braycurtis')
