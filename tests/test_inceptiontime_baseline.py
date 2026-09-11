import numpy as np
import pytest
import torch
from torch.nn import functional as F
import yaml
from pathlib import Path

from src.inceptiontime_baseline import InceptionNetwork, SameConv, fit_network
from src.neurotycho_learning import predict_binary


def protocol():
    return yaml.safe_load(Path('configs/analysis/inceptiontime-followup-260911.yaml').read_text())


def test_even_same_padding_and_binary_likelihood():
    convolution=SameConv(1,1,4,bias=False)
    with torch.no_grad():
        convolution.weight.copy_(torch.tensor([[[1.,2.,3.,4.]]]))
    x=torch.arange(5.).reshape(1,1,5)
    expected=F.conv1d(F.pad(x,(1,2)),convolution.weight)
    torch.testing.assert_close(convolution(x),expected)
    logits=torch.tensor([[2.,-1.],[-3.,1.],[.1,.2]])
    y=torch.tensor([0,1,1])
    torch.testing.assert_close(F.binary_cross_entropy_with_logits(logits[:,1]-logits[:,0],y.float()),
                               F.cross_entropy(logits,y))


def test_architecture_length_replay_and_channel_contract(tmp_path):
    torch.set_num_threads(2)
    config=protocol();model=InceptionNetwork(config['architecture']).eval()
    assert len(model.modules_list)==6 and len(model.shortcuts)==2
    assert all(isinstance(s[0],torch.nn.Conv1d) for s in model.shortcuts)
    assert [b.kernel_size[0] for b in model.modules_list[0].branches]==[40,20,10]
    x=torch.randn(2,117,16)
    before=predict_binary(model,x)
    assert before.shape==(2,)
    torch.save(model.state_dict(),tmp_path/'state.pt')
    restored=InceptionNetwork(config['architecture']).eval()
    restored.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True))
    np.testing.assert_array_equal(before,predict_binary(restored,x))
    with pytest.raises(ValueError):
        model(torch.randn(2,117,8))


def test_optimizer_can_fit_source_signal():
    torch.set_num_threads(2)
    config=protocol();torch.manual_seed(19)
    y=np.repeat([0.,1.],4)
    x=torch.randn(8,64,16)*.1+torch.tensor(y,dtype=torch.float32)[:,None,None]*2-1
    training={**config['training'],'batch_size':8}
    model,report=fit_network(x,y,None,config['architecture'],training,.001,0,11,epochs=20)
    assert report['history'][-1]['training_bce']<report['history'][0]['training_bce']*.5
    assert np.mean((predict_binary(model,x)>=.5)==y)==1
