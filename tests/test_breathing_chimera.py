import numpy as np
from scipy.integrate import solve_ivp
from scripts.scout_breathing_chimera import velocity,integrate,reduced,hopf,initial


def test_finite_rhs_matches_explicit_published_double_sum():
    theta=np.random.default_rng(1).normal(size=16);A=.28;alpha=np.pi/2-.1
    expected=np.ones(16)
    for i in range(16):
        for j in range(16):
            strength=(1+A)/2 if i//8==j//8 else (1-A)/2
            expected[i]+=strength/8*np.sin(theta[j]-theta[i]-alpha)
    np.testing.assert_allclose(velocity(theta,A),expected,atol=1e-14)


def test_hopf_fixed_point_has_zero_trace_and_positive_determinant():
    A,r,psi=hopf();s=np.array([r,psi]);eps=1e-6
    np.testing.assert_allclose(reduced(s,A),0,atol=1e-14)
    jac=np.column_stack([(reduced(s+eps*e,A)-reduced(s-eps*e,A))/(2*eps) for e in np.eye(2)])
    assert abs(np.trace(jac))<1e-8 and np.linalg.det(jac)>0
    assert .27<A<.29


def test_rk4_against_adaptive_solver_and_initial_condition():
    state=initial(16,0,'stratified');A=.28
    raw,r=integrate(state,A,.001,100,10,20)
    reference=solve_ivp(lambda t,s:velocity(s,A),(0,.4),state,t_eval=.1+.01*np.arange(1,21),
                        method='DOP853',rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(raw,np.sin(reference.y),atol=1e-9)
    assert r.shape==(2,20) and np.all(r<=1+1e-14)
