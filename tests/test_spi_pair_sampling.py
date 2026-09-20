import numpy as np
from scripts.scout_spi_pair_sampling import pair_vectors,correlation_features
from src.spi_spi_contract import build_unified_feature_values


def test_full_dyad_budget_matches_ordered_contract():
    a=np.random.default_rng(3).normal(size=(5,9,9))
    a[0]=(a[0]+a[0].T)/2
    mpis={str(k):v for k,v in enumerate(a)}
    gold,_,_=build_unified_feature_values(mpis,list(mpis))
    p=pair_vectors(a)
    np.testing.assert_allclose(correlation_features(p,np.arange(36)),gold,atol=1e-7)


def test_both_directions_preserved_and_common_transpose_invariant():
    a=np.random.default_rng(4).normal(size=(4,6,6))
    p=pair_vectors(a);s=np.array([1,5,8,10])
    np.testing.assert_array_equal(correlation_features(p,s),correlation_features(p[:,:,::-1],s))
    one=a.copy();one[0]=one[0].T
    assert not np.allclose(correlation_features(p,s),correlation_features(pair_vectors(one),s))


def test_missingness_is_not_silently_equated_to_full_validity():
    a=np.random.default_rng(5).normal(size=(3,4,4));a[0,0,1]=np.nan
    p=pair_vectors(a)
    full=correlation_features(p,np.arange(6))
    sparse=correlation_features(p,np.array([2,3,4,5]))
    assert np.isnan(full[:2]).all() and np.isfinite(sparse).all()
