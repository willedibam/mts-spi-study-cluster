from pathlib import Path
import numpy as np
import torch
import yaml
from src.neurotycho_learning import augment_source,date_state_weights,balanced_brier,fit_binary,predict_binary
from src.representation_state_neural import make_encoder


def test_date_state_mass_and_observation_augmentation():
    dates=np.array(['a']*6+['b']*4);y=np.array([0,0,1,1,1,1,0,1,1,1])
    w=date_state_weights(dates,y)
    masses=[w[(dates==date)&(y==label)].sum() for date in ['a','b'] for label in [0,1]]
    np.testing.assert_allclose(masses,[2.5]*4)
    x=torch.randn(2,2000,16);original=x.clone();rng=torch.Generator().manual_seed(71)
    shapes=set()
    for _ in range(20):
        out=augment_source(x,rng);shapes.add(tuple(out.shape[1:]))
        torch.testing.assert_close(out.mean(1),torch.zeros_like(out.mean(1)),atol=1e-6,rtol=0)
        torch.testing.assert_close(out.std(1,unbiased=False),torch.ones_like(out.std(1,unbiased=False)))
    assert shapes=={(1000,8),(1000,16),(2000,8),(2000,16)}
    torch.testing.assert_close(x,original)


def test_raw_encoder_channel_permutation_invariance():
    torch.set_num_threads(2)
    spec=yaml.safe_load(Path('configs/analysis/neurotycho-transfer-260910.yaml').read_text())['raw_encoder']
    model=make_encoder(spec).eval();x=torch.randn(2,1000,8)
    with torch.no_grad():
        expected=model(x);actual=model(x[:,:,torch.tensor([4,1,7,0,3,6,2,5])])
    torch.testing.assert_close(expected,actual,atol=2e-6,rtol=2e-6)


def test_binary_best_checkpoint_replays_validation_score(monkeypatch):
    class OneFeature(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(()))
        def forward(self,x):return self.weight*x.mean((1,2))
    monkeypatch.setattr('src.neurotycho_learning.make_encoder',lambda spec:OneFeature())
    y=np.array([0,1]*8);x=torch.tensor(2*y-1,dtype=torch.float32)[:,None,None]
    training=dict(maximum_epochs=30,minimum_epochs=5,patience=5,batch_size=8)
    model,report=fit_binary(x,y,np.array(['source']*len(y)),{},training,.1,0.,11,validation=(x,y))
    score=balanced_brier(y,predict_binary(model,x))
    assert abs(score-report['best_validation_brier'])<1e-8 and score<.05
