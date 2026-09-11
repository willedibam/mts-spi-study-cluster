import numpy as np

from src.neurotycho_library_baselines import FiniteColumns, channel_layout, fit_model, standardized


def test_training_missingness_filter_is_not_recomputed_on_target():
    x=np.column_stack([np.arange(20.),np.full(20,np.nan),np.ones(20)])
    model=FiniteColumns().fit(x)
    target=np.array([[np.nan,123.,7.],[40.,456.,8.]])
    np.testing.assert_array_equal(model.transform(target),[[9.5],[40.]])
    np.testing.assert_array_equal(model.finite_,[True,False,True])


def test_pooled_features_are_channel_permutation_invariant():
    x=np.random.default_rng(7).normal(size=(5,4,8));x[0,0,1]=np.nan
    np.testing.assert_allclose(channel_layout(x,'pooled'),channel_layout(x[:,[2,0,3,1]],'pooled'),atol=1e-15)


def test_rocket_bias_fitting_excludes_validation_animals(monkeypatch):
    import src.neurotycho_library_baselines as module
    fits=[]
    class ProbeRocket:
        def fit_transform(self,x):
            fits.append(x.copy());return self.transform(x)
        def transform(self,x):
            return x[:,0,:6]
    monkeypatch.setattr(module,'make_rocket',lambda *args:ProbeRocket())
    rng=np.random.default_rng(13)
    bank=dict(x=rng.normal(size=(18,2,30)),animal=np.repeat(['a','b','c'],6),
        archive=np.repeat(['a1','b1','c1'],6),y=np.tile([0,1],9),record_id=np.array([str(i) for i in range(18)]))
    config=dict(logistic_C=[.1],minimum_finite_fraction=.95,variance_threshold=1e-8,logistic_max_iter=2000)
    fit_model(bank,'minirocket',11,config)
    assert len(fits)==4
    normalized=standardized(bank['x'])
    for actual,group in zip(fits[:3],['a','b','c']):
        np.testing.assert_array_equal(actual,normalized[bank['animal']!=group])
    np.testing.assert_array_equal(fits[-1],normalized)
