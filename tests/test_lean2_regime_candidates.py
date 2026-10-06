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


def test_split_pde_against_independent_method_of_lines():
    n=32;L=16.;k=2*np.pi*np.fft.fftfreq(n,d=L/n);xx=np.arange(n)*L/n
    initial=1+.05*np.cos(2*np.pi*xx/L)+.03j*np.sin(4*np.pi*xx/L)
    rhs=lambda t,z:(1+3.5j)*np.fft.ifft(-k*k*np.fft.fft(z))+z-(1-.8j)*abs(z)**2*z
    reference=solve_ivp(rhs,(0,.2),initial,method='DOP853',rtol=1e-11,atol=1e-12).y[:,-1]
    errors=[]
    for dt in (.02,.01):
        a=initial.copy()
        for _ in range(round(.2/dt)):a=cgle_step(a,dt,3.5,.8,k)
        errors.append(np.max(abs(a-reference)))
    assert errors[1]<2e-6 and 3.8<errors[0]/errors[1]<4.2


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


def test_defect_charge_matches_global_winding_change():
    rng=np.random.default_rng(16)
    old=np.exp(1j*rng.uniform(-np.pi,np.pi,32));new=np.exp(1j*rng.uniform(-np.pi,np.pi,32))
    winding=lambda z: int(np.rint(np.angle(np.roll(z,-1)*z.conj()).sum()/(2*np.pi)))
    assert defect_charges(old,new).sum()==winding(old)-winding(new)


def test_bundle_preserves_raw_Q_and_separates_seeds(tmp_path):
    import json
    from scripts.lean2_candidate_readout import bundle
    raw=tmp_path/'physics';raw.mkdir();rng=np.random.default_rng(1)
    for i in range(4):
        np.savez_compressed(raw/f'case-{i:04d}.npz',observations=rng.normal(size=(16,1000)))
        (raw/f'case-{i:04d}.json').write_text(json.dumps(dict(system='rate',M=16,N=16,T=1000,seed=100+i,control=1.5,Q=i/10)))
    out=tmp_path/'bundle';bundle(raw,out,'test','/test',102)
    manifest=json.loads((out/'manifest.json').read_text())
    assert [r['role'] for r in manifest['rows']]==['development']*2+['evaluation']*2
    assert [r['Q'] for r in manifest['rows']]==[0.,.1,.2,.3]
    arrays=np.load(out/'observations.npz')
    for row in manifest['rows']:
        a=arrays[row['row_id']];np.testing.assert_allclose(a.mean(1),0,atol=1e-14)
        np.testing.assert_allclose(a.std(1),1,atol=1e-14)


def test_explicit_production_dimensions_and_completion_audit(tmp_path):
    import json
    import pytest
    from scripts.lean2_regime_candidates import run_case,audit_cases
    plan=tmp_path/'plan.json';indices=tmp_path/'indices.txt';out=tmp_path/'cases'
    plan.write_text(json.dumps({'tasks':[dict(system='rate',control=1.5,seed=100,N=16,M=16,T=32,burn=1.,reference=1.)]}))
    indices.write_text('0\n')
    with pytest.raises(FileNotFoundError):audit_cases(plan,indices,out)
    run_case(plan,0,out);audit_cases(plan,indices,out)
    row=json.loads((out/'case-0000.json').read_text());assert (row['M'],row['T'])==(16,32)


def test_figure_uses_T_column_not_dataframe_transpose(tmp_path,monkeypatch):
    import pandas as pd
    from matplotlib.figure import Figure
    from scripts.lean2_candidate_readout import figure
    seen=[]
    monkeypatch.setattr(Figure,'savefig',lambda self,*args,**kwargs:seen.append(self._suptitle.get_text()))
    scores=pd.DataFrame(dict(control=[1.3,1.4,1.5,1.6],Q=[-.03,-.01,.01,.03],z_PC1=[-2,-1,1,2],
        mean_PC1=[-1,-.5,.5,1],mean_abs_r=[.1,.2,.3,.4],role=['evaluation']*4,
        comparison_eligible=[True]*4,system=['rate']*4,M=[16]*4,N=[1024]*4,T=[1000]*4))
    figure(scores,tmp_path)
    assert all('T=1000' in title and len(title)<160 for title in seen)


def test_future_reference_does_not_change_observed_record():
    from scripts.lean2_regime_candidates import cgle
    for generator,params in [(rate,dict(control=1.5,N=16)),(cgle,dict(control=.8,L=16.))]:
        x,_,_=generator(seed=1,burn=1,T=32,reference=1.,**params)
        longer,_,_=generator(seed=1,burn=1,T=32,reference=2.,**params)
        np.testing.assert_array_equal(x,longer)


def test_published_driven_mean_field_boundary():
    from scripts.lean2_regime_candidates import rate_mean_field_boundary
    first=rate_mean_field_boundary(quadrature=128);second=rate_mean_field_boundary(quadrature=256)
    np.testing.assert_allclose(first,second,atol=1e-6)
    assert abs(first[0]-1.48)<.005


def test_secondary_diagnostic_reuses_coordinates(tmp_path,monkeypatch):
    import pandas as pd
    import scripts.lean2_candidate_readout as readout
    scores=pd.DataFrame(dict(Q=[0,.1,.2,.3],min_amplitude=[.8,.6,.01,.005],z_PC1=[-2,-1,1,2],
        z_standard_PC1=[-1,-.5,.5,1],mean_PC1=[-1,-.5,.5,1],mean_abs_r=[.4,.3,.2,.1],
        role=['evaluation']*4,comparison_eligible=[True]*4))
    original=scores.copy(deep=True);seen=[]
    monkeypatch.setattr(readout,'figure',lambda frame,*args,**kwargs:seen.append(frame.copy()))
    readout.secondary_amplitude(scores,tmp_path)
    pd.testing.assert_frame_equal(scores,original)
    np.testing.assert_array_equal(seen[0].z_PC1,-original.z_PC1)
    np.testing.assert_array_equal(seen[0].Q,original.min_amplitude)
    assert (tmp_path/'minimum-amplitude-metrics.csv').exists()
