import numpy as np
import pytest
from scripts.scout_cml2d_period_doubling import step, evolve, sensor_indices, order_summary, simulate, cases_from_config


def test_step_matches_roll_oracle_and_does_not_mutate_input():
    x=np.random.default_rng(13).random((11,11)); before=x.copy()
    mapped=np.empty_like(x); out=np.empty_like(x); r=3.86212; g=.2
    f=r*x*(1-x)
    truth=(1-4*g)*f+g*(np.roll(f,1,0)+np.roll(f,-1,0)+np.roll(f,1,1)+np.roll(f,-1,1))
    step(x,mapped,out,r,g)
    np.testing.assert_allclose(out,truth,rtol=0,atol=3e-16)
    np.testing.assert_array_equal(x,before)


def test_streaming_records_every_step_and_correct_sites():
    x=np.random.default_rng(14).random((8,8)); ids=sensor_indices(8,2)
    means,obs,last=evolve(x.copy(),3.85,.2,5,12,6,ids)
    for t in range(17):
        f=3.85*x*(1-x)
        x=.2*f+.2*(np.roll(f,1,0)+np.roll(f,-1,0)+np.roll(f,1,1)+np.roll(f,-1,1))
        if t>=5: np.testing.assert_allclose(means[t-5],x.mean(),atol=1e-13,rtol=0)
        if 5<=t<11: np.testing.assert_allclose(obs[t-5],x.ravel()[ids],atol=1e-12,rtol=0)
    np.testing.assert_allclose(last,x,atol=1e-12,rtol=0)


def test_order_measures_collective_alternation_and_future_only():
    means=np.tile([.3,.7],20)
    assert order_summary(means,8)['Q']==pytest.approx(.4)
    means[:8]=0
    assert order_summary(means,8)['Q']==pytest.approx(.4)
    assert order_summary(np.ones(40)*.5,8)['Q']==0


def test_sensor_layout_nesting_and_reproducibility():
    ids=sensor_indices(16,33)
    np.testing.assert_array_equal(ids,sensor_indices(16,33))
    for view in ids: assert len(np.unique(view))==64
    for n in (8,16,32,64):
        coords=np.stack(np.unravel_index(ids[1,:n],(16,16)),axis=1)
        assert len(np.unique(coords,axis=0))==n


def test_simulation_reproducible_and_grid():
    cfg=dict(g=.2,burn=8,record_steps=24,observation_steps=8)
    case=dict(L=8,r=3.85,seed=123)
    a,meta=simulate(case,cfg); b,_=simulate(case,cfg)
    for key in a: np.testing.assert_array_equal(a[key],b[key])
    assert meta['N']==64 and a['observed'].dtype==np.float64
    assert len(cases_from_config({'grid':{'L':[8,16],'r':[3.8,3.9]},'cases':[case]}))==5
    with pytest.raises(ValueError): simulate({**case,'g':.3},cfg)
