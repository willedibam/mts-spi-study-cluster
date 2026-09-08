import torch
import pytest
from src.representation_state_neural import make_encoder


@pytest.mark.parametrize('architecture',['pair_relation','pointwise_pair'])
def test_pair_relation_symmetry_variable_shape_and_gradients(architecture):
    torch.set_num_threads(2)
    torch.manual_seed(32)
    model=make_encoder({'architecture':architecture}).eval()
    x=torch.randn(2,128,8)
    a=model(x);b=model(x[:,:,[3,0,6,2,5,7,1,4]])
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    assert model(x[:,:64,:4]).shape==(2,)
    a.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
