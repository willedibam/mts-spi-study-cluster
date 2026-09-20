import hashlib
import json
import numpy as np
import pytest
import yaml
from scripts.scout_cml2d_period_doubling import simulate, step
from scripts.prepare_cml2d_corpus import prepare


def test_affine_conjugacy_preserves_full_2d_update():
    r=3.86212;g=.2;alpha=r*(r-2)/4
    x=np.random.default_rng(2).random((9,9));u=(4*x-2)/(r-2)
    out=np.empty_like(x);step(x,np.empty_like(x),out,r,g)
    f=1-alpha*u*u
    truth=(1-4*g)*f+g*(np.roll(f,1,0)+np.roll(f,-1,0)+np.roll(f,1,1)+np.roll(f,-1,1))
    np.testing.assert_allclose((4*out-2)/(r-2),truth,atol=1e-15,rtol=0)


def test_export_is_nested_targets_are_disjoint_and_hash_bound(tmp_path):
    source=tmp_path/'masters';source.mkdir()
    arrays,meta=simulate(dict(L=8,r=3.85,seed=21),dict(g=.2,burn=8,record_steps=40,observation_steps=16))
    np.savez_compressed(source/'case-000.npz',**arrays,metadata_json=json.dumps(meta))
    out=tmp_path/'corpus';config=tmp_path/'corpus.yaml'
    prepare(source,out,config,[8,16],[4,8],['dispersed','contiguous'],[21])
    manifest=json.loads((out/'manifest.json').read_text());spec=yaml.safe_load(config.read_text())
    assert len(manifest['rows'])==8 and all(row['role']=='development' for row in manifest['rows'])
    assert spec['source']['sha256']==hashlib.sha256((out/'observations.npz').read_bytes()).hexdigest()
    with np.load(out/'observations.npz') as a:
        for row in manifest['rows']:
            view=meta['views'].index(row['view'])
            np.testing.assert_array_equal(a[row['row_id']],arrays['observed'][:row['T'],view,:row['M']].T)
            assert row['Q_reference']==meta['Q']
            assert row['Q_window']==pytest.approx(np.abs(np.diff(arrays['global_mean'][:row['T']].reshape(-1,2),axis=1)).mean())
    with pytest.raises(FileExistsError):prepare(source,out,config,[8],[4],['dispersed'],[21])
    sensitivity=tmp_path/'sensitivity'
    prepare(source,sensitivity,tmp_path/'sensitivity.yaml',[8,16],[4,8],['dispersed'],[21],exclude_shapes=[(16,8)])
    rows=json.loads((sensitivity/'manifest.json').read_text())['rows']
    assert len(rows)==3 and {(row['M'],row['T']) for row in rows}=={(8,4),(8,8),(16,4)}
    selected=[]
    for path in sensitivity.glob('indices-*.txt'):
        selected.extend(map(int,path.read_text().split()))
    assert sorted(selected)==[1,2,3]
