import numpy as np
from scripts.confirm_normal_margin_copula import combined_rows
from scripts.report_dependence_transition import coordinate


def test_old_evaluation_never_enters_confirmation():
    old=[dict(row_id='old-fit',role='development'),dict(row_id='old-held',role='evaluation')]
    new=[dict(row_id='new-fit',role='development'),dict(row_id='new-held',role='evaluation')]
    rows=combined_rows(old,new)
    assert [r['row_id'] for r in rows]==['old-fit','new-fit','new-held']
    assert [r['original_development'] for r in rows]==[True,False,False]


def test_frozen_projection_independent_of_new_rows():
    rng=np.random.default_rng(12);x=rng.normal(size=(20,7));fit=np.arange(20)<10
    q,evr,_=coordinate(x,fit,False)
    augmented=np.concatenate([x,rng.normal(size=(30,7))]);fit2=np.r_[fit,np.zeros(30,dtype=bool)]
    q2,evr2,_=coordinate(augmented,fit2,False)
    np.testing.assert_allclose(q,q2[:20],atol=1e-12);assert evr==evr2
