import numpy as np
import pytest
from scripts.calibrate_all_proof_coupling import kuramoto_gain, inventory, EXTRA, simulate
from src.generators import generate_wave_1d, generate_kuramoto


def test_phase_gain_matches_actual_update_finite_differences():
    theta=np.random.default_rng(1).uniform(0,2*np.pi,size=16);K=-53.;dt=.00625
    def update(x):
        diff=x[None,:]-x[:,None]
        return x+dt*K*np.sin(diff).sum(axis=1)/15
    eps=1e-6;eye=np.eye(16)
    jac=np.array([(update(theta+eps*e)-update(theta-eps*e))/(2*eps) for e in eye]).T
    np.fill_diagonal(jac,0)
    np.testing.assert_allclose(kuramoto_gain(theta[None],K),np.abs(jac).sum(axis=1).mean(),rtol=1e-8)


def test_phase_sine_view_matches_original_observation():
    case=next(c for c in EXTRA if c['label']=='Kuramoto-fast')
    x,_=simulate(case,4.,7,t=20)
    direct=generate_kuramoto(M=16,T=20,dt=.00625,K=-4.,omega_mean=3.,omega_std=1.73205,
        eta=0,output='sin',connectivity='all-to-all',transients=2000,zscore=False,rng=np.random.default_rng(7))
    np.testing.assert_array_equal(x,direct)


def test_wave_fixed_timestep_preserves_default_and_changes_coupling():
    kwargs=dict(M=16,T=100,n_modes=5,noise_std=0,zscore=False)
    default=generate_wave_1d(**kwargs,c=10,seed=8)
    explicit=generate_wave_1d(**kwargs,c=10,dt=.00125,seed=8)
    np.testing.assert_array_equal(default,explicit)
    x=generate_wave_1d(**kwargs,c=10*np.sqrt(2.5),dt=.00125,seed=8)
    lap=np.roll(x[1:-1],1,axis=1)+np.roll(x[1:-1],-1,axis=1)-2*x[1:-1]
    np.testing.assert_allclose(x[2:],2*x[1:-1]-x[:-2]+.1*lap,atol=1e-14)
    assert not np.allclose(x,default)
    with pytest.raises(ValueError):generate_wave_1d(**kwargs,c=100,dt=.1)


def test_all_historical_classes_covered_and_fresh_duplicate_disclosed():
    inv=inventory()
    assert len(inv.query("source=='historical14'"))==14
    assert inv.normalized.nunique()==16
    assert inv.query("source=='fresh_VAR'").normalized.nunique()==2
