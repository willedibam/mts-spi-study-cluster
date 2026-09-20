import numpy as np
import pytest
from scripts.scout_cml2d_period_doubling import step
from scripts.cml2d_structure_diagnostic import (
    structural_cases,coupling_matrix,tangent_step,lyapunov_blocks,orbit_diagnostics,recurrence_period)


def test_design_counts_and_original_indices():
    cases=structural_cases()
    assert len(cases)==620
    assert sum(c['preparation']=='random' for c in cases)==440
    assert sum('original_index' in c for c in cases)==36
    assert all(0<=c['original_index']<1360 for c in cases if 'original_index' in c)
    assert len({tuple(sorted(c.items())) for c in cases})==620


def test_tangent_matches_dense_jacobian_and_finite_difference():
    rng=np.random.default_rng(14)
    x=rng.random((6,6));v=rng.normal(size=x.shape);r=3.86212;g=.2
    analytic=tangent_step(x,v,r,g)
    dense=coupling_matrix(6,g)@(r*(1-2*x.ravel())*v.ravel())
    np.testing.assert_allclose(analytic.ravel(),dense,atol=1e-14)
    eps=1e-6
    plus=np.empty_like(x);minus=np.empty_like(x)
    step(x+eps*v,np.empty_like(x),plus,r,g)
    step(x-eps*v,np.empty_like(x),minus,r,g)
    np.testing.assert_allclose(analytic,(plus-minus)/(2*eps),atol=2e-10)


def test_fixed_point_floquet_and_lyapunov_oracle():
    r=2.8;state=np.full((5,5),1-1/r)
    result=orbit_diagnostics(np.tile(state.ravel(),(100,1)),r)
    assert result['micro_period']==1 and result['constant_channels']==25
    assert result['floquet_radius']==pytest.approx(abs(2-r))
    tangent=np.random.default_rng(4).normal(size=state.shape)
    blocks=lyapunov_blocks(state,tangent,r,.2,alignment=1000,steps=1000)
    np.testing.assert_allclose(blocks,np.log(abs(2-r)),atol=1e-12)


def test_recurrence_detects_full_period_not_just_powers_of_two():
    assert recurrence_period(np.tile([.1,.2,.3],100))==3
    assert recurrence_period(np.random.default_rng(4).random(300))==0
