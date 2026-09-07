import copy
import json
from pathlib import Path
import numpy as np
import pytest
import yaml

from src.interaction_share_learning import select_statistical, select_reference
from src.interaction_share_learning import standardized_marginal_shapes
from src.representation_state_data import file_hash
from scripts.run_representation_state_pilot import run


def test_marginal_shapes_match_positive_affine_invariance_and_ignore_alignment():
    from src.representation_attribution import rich_marginals
    from src.mpi_representation_baselines import summarize_mpis
    rng=np.random.default_rng(22)
    mpis={'a':rng.normal(size=(6,6)), 'b':rng.exponential(size=(6,6)), 'constant':np.ones((6,6))}
    order=list(mpis)
    def shape(values):
        valid=summarize_mpis(values,order)[2][None]
        return standardized_marginal_shapes(rich_marginals(values,order)[None],valid)
    transformed={k:scale*mpis[k]+shift for k,scale,shift in zip(order,[2.,.3,4.],[7.,-1.,2.])}
    np.testing.assert_allclose(shape(mpis),shape(transformed),atol=1e-12,equal_nan=True)
    mask=~np.eye(6,dtype=bool)
    permuted={k:v.copy() for k,v in mpis.items()}
    for value in permuted.values(): value[mask]=rng.permutation(value[mask])
    np.testing.assert_allclose(shape(mpis),shape(permuted),atol=1e-12,equal_nan=True)
    assert np.isnan(shape(mpis)[0,-21:]).all()


@pytest.mark.parametrize('head',['pca','pls','rbf'])
def test_selection_isolated_from_nontraining_records(head):
    methods=yaml.safe_load(Path('configs/analysis/interaction-share-260908.yaml').read_text())['methods']
    rng=np.random.default_rng(11)
    bank={'m':rng.normal(size=(30,9)),'z':rng.normal(size=(30,14))}
    targets=rng.uniform(size=30); strata=np.tile(np.arange(5),6); train=np.arange(20)
    chosen,log=select_statistical(bank,'m+z',train,targets,strata,methods,11,head)
    changed=copy.deepcopy(bank)
    for x in changed.values(): x[20:]=np.nan
    targets[20:]=-1e9
    again,newlog=select_statistical(changed,'m+z',train,targets,strata,methods,11,head)
    assert chosen==again and log==newlog
    for fold in log['folds']:
        assert set(fold['fit']).isdisjoint(fold['validation'])
        assert set(fold['fit']+fold['validation'])==set(train)


def test_raw_regularization_selection_cannot_use_held_labels_or_values():
    rng=np.random.default_rng(7); x=rng.uniform(size=(30,4)); y=rng.uniform(size=30)
    strata=np.tile(np.arange(5),6); train=np.arange(20)
    choice,model,log=select_reference(x,train,y,strata,11)
    before=model.predict(x[train,choice])
    x[20:]=1e9;y[20:]=-1e9
    again,model2,log2=select_reference(x,train,y,strata,11)
    assert choice==again and log==log2
    np.testing.assert_array_equal(before,model2.predict(x[train,choice]))


def test_shared_runner_filters_training_family_and_keeps_both_test_families(tmp_path):
    from scripts.run_representation_state_random_control import run as run_random
    p=yaml.safe_load(Path('configs/analysis/interaction-share-260908.yaml').read_text())
    p['methods']['subset_seeds']=[11];p['sampling']['labelled_training_masters_per_coupling']=[2]
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(p))
    data=tmp_path/'data';data.mkdir()
    rows=[]
    for role in ['training_pool','evaluation']:
        for family in ['linear','tanh']:
            for k in range(5):
                for r in range(2):
                    i=len(rows)
                    rows.append(dict(row_id=str(i),master_id=str(i),master_index=i,role=role,family=family,coupling_index=k,M=4,T=16,target=.2 if family=='linear' else .8))
    np.save(data/'masters.npy',np.zeros((40,16,4)))
    np.save(data/'observables.npy',np.zeros((40,1)))
    (data/'manifest.json').write_text(json.dumps(dict(config_sha256=file_hash(config),rows=rows,artifacts={name:file_hash(data/name) for name in ['masters.npy','observables.npy']})))
    run(config,data,tmp_path/'fits',['mean'],'cpu',source_family='linear')
    with np.load(tmp_path/'fits/mean-n2-s11.npz') as a:
        assert all(rows[i]['family']=='linear' and rows[i]['role']=='training_pool' for i in a['train_indices'])
        assert {rows[i]['family'] for i in a['evaluation_indices']}=={'linear','tanh'}
        np.testing.assert_allclose(a['prediction'],.2)
    run_random(config,data,tmp_path/'random','cpu',source_family='linear')
    with np.load(tmp_path/'random/random_encoder-n2-s11.npz') as a:
        assert all(rows[i]['family']=='linear' and rows[i]['role']=='training_pool' for i in a['train_indices'])
        np.testing.assert_allclose(a['prediction'],.2)
