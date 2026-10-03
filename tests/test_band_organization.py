import numpy as np
from scripts.band_organization_experiment import covariance_layers, frequency_weights, simulate, CLASSES
from scripts.audit_spi_strength import normalized_quantiles


def test_band_permutations_preserve_covariance_and_marginals_but_not_correspondence():
    mask=~np.eye(16,dtype=bool)
    for strength in [.35,.55,.75]:
        base=covariance_layers(strength)
        signatures=[]
        for label in CLASSES:
            layers=base[['ABC'.index(c) for c in label]]
            np.testing.assert_allclose(layers.mean(axis=0),base.mean(axis=0))
            np.testing.assert_allclose(np.diagonal(layers,axis1=1,axis2=2),1)
            for layer in layers:
                np.testing.assert_array_equal(np.sort(layer[mask]),np.sort(base[0][mask]))
            signatures.append(np.corrcoef(layers[:,mask])[np.triu_indices(3,1)])
        assert len(np.unique(np.round(signatures,12),axis=0))==6


def test_disjoint_spectral_support_and_exact_population_variance():
    w=frequency_weights(1000)
    assert ((w>0).sum(axis=0)<=1).all()
    np.testing.assert_allclose(2*np.sum(w*w,axis=1)/1000,1)
    x,meta=simulate('ABC',2);y,other=simulate('ABC',2)
    np.testing.assert_array_equal(x,y)
    assert meta==other and x.shape==(1000,16)
    assert np.isfinite(x).all()


def test_binary_swap_preserves_broad_band_controls_and_has_declared_signature():
    layers=covariance_layers(.55)/3
    first,second=layers[[1,2,0]],layers[[2,1,0]]
    np.testing.assert_allclose(first.sum(axis=0),second.sum(axis=0))
    np.testing.assert_array_equal(np.maximum(first[0],first[1]),np.maximum(second[0],second[1]))
    np.testing.assert_array_equal(first[2],second[2])
    mask=~np.eye(16,dtype=bool)
    for matrices,expected in [(first,[-1/24,-1/4,3/8]),(second,[-1/24,3/8,-1/4])]:
        np.testing.assert_allclose(matrices[:,mask].mean(axis=1),.55/15)
        np.testing.assert_allclose(np.corrcoef(matrices[:,mask])[np.triu_indices(3,1)],expected)


def test_normalized_quantiles_equal_direct_edge_normalization():
    rng=np.random.default_rng(3);v=rng.normal(size=(2,3,30))
    summary=np.concatenate([v.mean(axis=2)[:,:,None],v.std(axis=2)[:,:,None],
        np.quantile(v,[.1,.25,.5,.75,.9],axis=2).transpose(1,2,0)],axis=2)
    shape,valid=normalized_quantiles(summary)
    normalized=(v-v.mean(axis=2,keepdims=True))/v.std(axis=2,keepdims=True)
    expected=np.quantile(normalized,[.1,.25,.5,.75,.9],axis=2).transpose(1,2,0).reshape(2,-1)
    np.testing.assert_allclose(shape,expected,atol=1e-14)
    assert valid.all()
    np.testing.assert_allclose(np.corrcoef(v[0]),np.corrcoef(normalized[0]),atol=1e-14)
