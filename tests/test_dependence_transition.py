import numpy as np
from scripts.scout_dependence_transition import rhs, maps, normalize, probes, integrate


def test_response_tangent_matches_finite_difference():
    s=np.array([1.,2.,.2, 3.,-2.,4., 1.,1.,2., .3,-.2,.5])
    for hetero in (False,True):
        d=rhs(s,.12,1.3,hetero)
        delta=np.zeros(12);delta[3:6]=1e-6*s[9:12]
        fd=(rhs(s+delta,.12,1.3,hetero)[3:6]-rhs(s-delta,.12,1.3,hetero)[3:6])/2e-6
        np.testing.assert_allclose(d[9:12],fd,atol=1e-8)


def test_logistic_complete_sync_threshold_and_auxiliary_exclusion():
    init=np.random.default_rng(42).uniform(.05,.95,(3,4))
    x,cle,aux,identical=maps(init,.6,10000,20000)
    assert x.shape==(8,1000)
    np.testing.assert_allclose(cle,np.log(.8),atol=.002)
    assert np.max(aux)<1e-8 and np.max(identical)<1e-8
    x0,cle0,_,_=maps(init,0.,10000,20000)
    np.testing.assert_allclose(cle0,np.log(2),atol=.002)
    np.testing.assert_array_equal(x[::2],x0[::2])


def test_probes_and_normalization_are_deterministic():
    x=np.random.default_rng(8).normal(size=(6,1000))
    m,z,names=probes(x)
    assert m.shape==(11,) and z.shape==(55,) and len(names)==11
    assert np.isfinite(m).all() and np.isfinite(z).all()
    np.testing.assert_allclose(normalize(x).std(axis=1),1,atol=1e-14)


def test_rk4_observations_match_independent_adaptive_integrator():
    from scipy.integrate import solve_ivp
    s=np.array([1.,2.,.2, 3.,-2.,4., 1.,1.,2., .3,-.2,.5])
    for hetero,k in [(False,.11),(True,6.6)]:
        x,_,_=integrate(s,k,1.3,hetero,.001,10,100,100,20)
        times=.1+.01*np.arange(1,21)
        reference=solve_ivp(lambda t,y:rhs(y,k,1.3,hetero),(0,.4),s,
                            method='DOP853',t_eval=times,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(x,reference.y[:6],rtol=2e-6,atol=2e-7)
