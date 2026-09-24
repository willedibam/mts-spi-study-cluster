"""The displayed outcomes must not influence feature choice or AUC orientation."""
import numpy as np

from scripts.refresh_proof_figures import rank_pair, preprocess


def test_raincloud_selection_and_orientation_ignore_display_rows():
    instances=np.tile(np.arange(10),2)
    labels=np.repeat(['a','b'],10)
    x=np.zeros((20,2));x[10:16,0]=1
    # Feature 1 separates display rows perfectly but ties on training rows.
    x[16:,1]=100
    # The selected feature reverses on display: report AUC=0, never reorient.
    x[6:10,0]=1
    bank={'instance':instances,'y':labels,'X_sym':x,'pairs':np.array([['x','y'],['x','z']])}
    row=rank_pair(bank,'a','b')
    assert row['feature_index']==0
    assert row['training_auc_oriented']==1
    assert row['display_auc_fixed_orientation']==0


def test_preprocessing_retains_rows_and_filters_unusable_coordinates():
    x=np.column_stack([np.arange(20,dtype=float),np.ones(20),np.full(20,np.nan)])
    x[0,0]=np.nan
    y,keep=preprocess(x)
    assert keep.tolist()==[True,False,False]
    assert y.shape==(20,1) and np.isfinite(y).all()
    assert y[0,0]==10
