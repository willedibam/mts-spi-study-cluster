import numpy as np
from scripts.plot_large_m_pair_sampling import metrics, family, sampling_budgets


def test_error_does_not_hide_undefined_reference_coordinates():
    gold=np.array([.2,.7,np.nan])
    sampled=np.array([[.2,np.nan,.5],[.1,.5,.6]])
    result=metrics(sampled,gold)
    np.testing.assert_allclose(result['retained'],[.5,1])
    assert result['rmse'][0]==0 and np.isnan(result['strict_rmse'][0])
    np.testing.assert_allclose(result['strict_rmse'][1],np.sqrt((.1**2+.2**2)/2))
    np.testing.assert_allclose(result['coordinate_valid_fraction'],[1,.5,0])
    np.testing.assert_allclose(result['apparent_validity'],[1,1])


def test_census_has_zero_error_and_full_valid_reference_coverage():
    gold=np.array([.2,.7,np.nan])
    result=metrics(np.stack([gold,gold]),gold)
    np.testing.assert_array_equal(result['rmse'],[0,0])
    np.testing.assert_array_equal(result['retained'],[1,1])
    np.testing.assert_array_equal(result['strict_rmse'],[0,0])


def test_families_keep_sparse_and_dense_var_separate():
    assert family('var-sparse-s1-M100')=='VAR sparse'
    assert family('var-dense-s1-M100')=='VAR dense'


def test_large_budgets_extend_without_changing_existing_small_budgets():
    for m in [64,100,243,256]:
        d=m*(m-1)//2
        b=sampling_budgets(d)
        assert b[-1]==d and (np.diff(b)>0).all()
        old=np.array(sorted({min(d,n) for n in [8,16,32,50,100,200,400,800,1600,3200,d]}))
        if m<=100:
            np.testing.assert_array_equal(b,old)
        else:
            assert {6400,12800,25600}.issubset(b)


def test_single_exceptional_orientation_can_determine_feature_validity():
    from scripts.scout_spi_pair_sampling import correlation_features
    pairs = np.ones((2, 3, 2))
    pairs[0, 2, 1] = .6
    pairs[1] = np.arange(6).reshape(3, 2)
    assert np.isnan(correlation_features(pairs, np.array([0, 1]))[0])
    assert np.isfinite(correlation_features(pairs, np.arange(3))[0])
