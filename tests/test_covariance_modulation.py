import numpy as np
from src.covariance_modulation import simulate, population_covariance, raw_references


def test_population_covariance_and_channel_laws():
    d=np.linspace(.25,.6,8); c=.1
    base=population_covariance(c,d)
    for alpha in [0,.3,1]:
        plus=population_covariance(c,d,alpha,1)
        minus=population_covariance(c,d,alpha,-1)
        np.testing.assert_allclose((plus+minus)/2,base)
        np.testing.assert_allclose(np.diag(plus),1)
        assert np.linalg.eigvalsh(plus).min()>0
        # Independent Gaussian innovations at different times and unit
        # conditional diagonal establish each channel's iid N(0,1) law.


def test_simulator_moments_and_replay():
    d=np.full(4,.5)
    x,s=simulate(.8,.1,d,'iid',32,t=200000)
    np.testing.assert_allclose(x.T@x/len(x),population_covariance(.1,d),atol=.015)
    fourth=np.mean(x[:,0]**2*x[:,2]**2)-1-2*.1**2
    assert abs(fourth-2*.8**2*.5**2)<.035
    a,sa=simulate(.5,.1,d,'persistent',64,t=100)
    b,sb=simulate(.5,.1,d,'persistent',64,t=100)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(sa,sb)


def test_reference_permutation_invariance():
    x,_=simulate(.7,.1,np.linspace(.3,.6,8),'persistent',4)
    first=raw_references(x)
    second=raw_references(x[:,[3,0,6,2,5,7,1,4]])
    for key in first: np.testing.assert_allclose(first[key],second[key],atol=1e-10)
