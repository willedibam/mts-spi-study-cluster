import numpy as np
import torch
from torch import nn
from src.interaction_share_learning import fit_statistical
from src.representation_state_neural import fit_encoder


def test_pls_caps_rank_of_duplicate_validity_columns():
    x=np.tile(np.array([0.,0.,1.,1.,1.])[:,None],(1,6))
    config=dict(minimum_valid_fraction=.95,variance_threshold=1e-8,z_scaling='center',clip_standard_deviations=5.)
    transform,model=fit_statistical({'validity':x},'validity',np.arange(5),
                                    np.array([.1,.3,.4,.6,.9]),config,'pls',4)
    assert model.n_components==1
    assert np.isfinite(model.predict(transform.transform({'validity':x},np.arange(5)))).all()


class ConstantPrediction(nn.Module):
    def __init__(self):
        super().__init__();self.bias=nn.Parameter(torch.tensor(-1.))
    def forward(self,x):return self.bias.expand(len(x))


def test_unclipped_stopping_recognizes_improvement_below_zero(monkeypatch):
    monkeypatch.setattr('src.representation_state_neural.make_encoder',lambda spec:ConstantPrediction())
    x=torch.zeros(4,4,2);y=torch.full((4,),.5)
    spec=dict(batch_size=2,maximum_epochs=8,minimum_epochs=3,early_stopping_patience=2)
    _,old=fit_encoder(x,y,spec,.01,0.,3,validation=(x,y))
    _,new=fit_encoder(x,y,{**spec,'early_stopping_metric':'validation_unclipped_MAE'},.01,0.,3,validation=(x,y))
    assert old['best_epoch']==1 and old['epochs_run']==3
    assert new['best_epoch']==8 and new['epochs_run']==8
    assert new['validation_MAE']==old['validation_MAE']==.5
    assert new['history'][-1]['validation_unclipped_MAE']<new['history'][0]['validation_unclipped_MAE']
