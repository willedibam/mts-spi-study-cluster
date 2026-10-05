import numpy as np
from scripts.confirm_normal_margin_copula import combined_rows,frozen_coordinate
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


def test_row_ids_are_pickle_free(tmp_path):
    import pandas as pd
    frame=pd.DataFrame(combined_rows([dict(row_id='old-fit',role='development')],
                                   [dict(row_id='new-held',role='evaluation')]))
    np.savez_compressed(tmp_path/'ids.npz',row_id=frame.row_id.to_numpy(dtype=str))
    with np.load(tmp_path/'ids.npz',allow_pickle=False) as bank:
        np.testing.assert_array_equal(bank['row_id'],['old-fit','new-held'])


def test_frozen_preprocessing_replays_float32_original_for_both_scalings():
    rng=np.random.default_rng(17);x=rng.normal(size=(24,40)).astype(np.float32);x[:,0]*=1e-6
    fit=np.arange(24)<12;new=rng.normal(size=(50,40)).astype(np.float32);new[:,0]*=1e-6
    for standard in [False,True]:
        q,evr,_=coordinate(x,fit,standard)
        old,new_q,evr2,_=frozen_coordinate(x,new,fit,standard)
        np.testing.assert_allclose(old,q,atol=1e-7);assert evr==evr2
        _,new_q2,_,_=frozen_coordinate(x,new*10,fit,standard)
        assert not np.allclose(new_q,new_q2)
