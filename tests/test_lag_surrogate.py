import numpy as np
import pytest
from scripts.build_lag_surrogate import pair

@pytest.mark.parametrize('m,t',[(8,100),(16,500),(32,1000)])
def test_surrogate_preserves_circular_second_order(m,t):
    (x,y),_=pair(np.random.default_rng(123),m=m,t=t)
    assert not np.allclose(x,y)
    for lag in [0,1,5,10,t-1]:
        np.testing.assert_allclose(x@np.roll(x,lag,axis=1).T,y@np.roll(y,lag,axis=1).T,atol=1e-10)
    np.testing.assert_allclose(abs(np.fft.rfft(x)),abs(np.fft.rfft(y)),atol=1e-10)


def test_seed_replays_null_and_signal():
    for strength in [0.,1.]:
        (x,y),_=pair(np.random.default_rng(321),strength=strength)
        (a,b),_=pair(np.random.default_rng(321),strength=strength)
        np.testing.assert_array_equal(x,a)
        np.testing.assert_array_equal(y,b)
