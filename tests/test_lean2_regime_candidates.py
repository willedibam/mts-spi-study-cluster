import numpy as np
from scipy.integrate import solve_ivp
from scripts.lean2_regime_candidates import cgle_step,local_flow,defect_charges,rate


def test_local_flow_against_independent_ode():
    a=np.array([.2+.4j,1.2-.1j]);c=.8;h=.17
    exact=solve_ivp(lambda t,z:z-(1-1j*c)*abs(z)**2*z,(0,h),a,rtol=1e-11,atol=1e-12).y[:,-1]
    np.testing.assert_allclose(local_flow(a,h,c),exact,rtol=1e-10,atol=1e-11)


def test_plane_wave_second_order_convergence():
    n=64;L=32.;k=2*np.pi*np.fft.fftfreq(n,d=L/n);wave=2*np.pi/L
    initial=np.sqrt(1-wave**2)*np.exp(1j*wave*np.arange(n)*L/n)
    exact=initial*np.exp(1j*(.8-(3.5+.8)*wave**2)*2)
    errors=[]
    for dt in (.04,.02,.01):
        a=initial.copy()
        for _ in range(round(2/dt)):a=cgle_step(a,dt,3.5,.8,k)
        errors.append(np.max(abs(a-exact)))
    assert 3.9<errors[0]/errors[1]<4.1 and 3.9<errors[1]/errors[2]<4.1


def test_defect_counts_local_and_canceling_events():
    # Explicit vortex and antivortex on distinct plaquettes; net winding can cancel.
    old=np.exp(1j*np.array([-.75,-.25,.25,.75,.25,-.25])*np.pi)
    new=old.conj()
    charges=defect_charges(old,new)
    assert abs(charges).sum()==4 and charges.sum()==0
    assert not defect_charges(old,old).any()


def test_uncoupled_ou_and_tangent():
    x,q,_=rate(0.,0,N=16,dt=.02,burn=100,reference=600,T=1000,sample=.5)
    assert abs(q['Q']-np.log(1-.02+.5*.02**2)/.02)<1e-10
    assert abs(q['state_variance']-.125)<.015
    assert np.linalg.matrix_rank(x)==16
