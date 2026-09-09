import numpy as np
import torch
from src.spi_edge_pool import SPIEdgePool,standardize_edges,pack_inputs
from src.spi_spi_contract import build_unified_feature_values
from src.representation_state_neural import fit_encoder,predict


def test_edges_recover_pearson_and_invalid_contract():
    rng=np.random.default_rng(8);mpis={str(i):rng.normal(size=(8,8)) for i in range(5)}
    mpis['3'][:]=1;mpis['4'][0,1]=np.nan
    order=list(mpis);x,valid=standardize_edges(mpis,order)
    assert valid.tolist()==[True,True,True,False,False]
    assert np.isfinite(x).all()
    z=build_unified_feature_values(mpis,order)[0]
    gram=x.T@x/len(x);gram[~valid,:]=np.nan;gram[:,~valid]=np.nan
    np.testing.assert_allclose(gram[np.triu_indices(5,1)],z,atol=2e-7,equal_nan=True)
    changed={k:3*v+7 for k,v in mpis.items()};xx,vv=standardize_edges(changed,order)
    np.testing.assert_allclose(xx,x,atol=1e-6);np.testing.assert_array_equal(vv,valid)


def test_edge_pool_invariant_and_variable_size():
    torch.manual_seed(9);model=SPIEdgePool(dict(input_width=8,edge_width=32,head_width=32))
    x=torch.randn(3,56,8);y=model(x)
    torch.testing.assert_close(y,model(x[:,torch.randperm(56)]))
    torch.testing.assert_close(y,model(x.repeat(1,2,1)))
    model(torch.randn(3,240,8)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_learns_planted_cross_statistic_agreement():
    torch.set_num_threads(2);rng=np.random.default_rng(17);n=80;e=56
    y=np.tile([0.,1.],n//2).astype(np.float32);a=rng.normal(size=(n,e));noise=rng.normal(size=(n,e))
    b=(2*y[:,None]-1)*.85*a+np.sqrt(1-.85**2)*noise
    values=np.stack([a,b],axis=-1);values=(values-values.mean(1,keepdims=True))/values.std(1,keepdims=True)
    x=torch.tensor(pack_inputs(values,np.ones((n,2),bool)))
    spec=dict(architecture='spi_edge_pool',input_width=4,edge_width=32,head_width=32,
        batch_size=16,maximum_epochs=150,minimum_epochs=30,early_stopping_patience=20,
        early_stopping_metric='validation_unclipped_MAE')
    model,_=fit_encoder(x[:40],torch.tensor(y[:40]),spec,.001,.0001,11,epochs=150)
    assert ((predict(model,x[40:],16)>=.5)==y[40:]).mean()>=.95
