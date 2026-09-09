"""Apply source-selected models to the prospective faster-regime cohort.

Statistical models are reconstructed from source labels and saved settings, with
an original-test prediction replay before any target prediction. Neural weights
are frozen checkpoints. No target label enters fitting or selection.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch,yaml
from src.interaction_share_learning import fit_statistical,standardized_marginal_shapes
from src.representation_state_data import file_hash,observed_view
from src.representation_state_neural import make_encoder,predict
from src.spi_edge_pool import pack_inputs
from src.run_external_corpus import _atomic_json,_atomic_savez


def main(config,source_data,source_root,target_data,target_bank,edge_bank,output):
    cfg=yaml.safe_load(config.read_text());sr=json.loads((source_data/'manifest.json').read_text())['rows']
    tr=json.loads((target_data/'manifest.json').read_text())['rows'];n=len(sr);y=np.array([r['target'] for r in sr])
    banks=[]
    order=None
    for p,data,rows in [(source_root/'gadi-analysis/features.npz',source_data,sr),(target_bank,target_data,tr)]:
        assert file_hash(p)==json.loads(p.with_suffix('.json').read_text())['artifact_sha256']
        with np.load(p,allow_pickle=False) as a:
            assert a['manifest_sha256'].item()==file_hash(data/'manifest.json')
            np.testing.assert_array_equal(a['row_id'],[r['row_id'] for r in rows])
            if order is not None:np.testing.assert_array_equal(a['spi_order'],order)
            order=a['spi_order'];banks.append({k:a['X_'+k] for k in ['m','g','z','validity']})
    bank={k:np.concatenate([a[k] for a in banks]) for k in banks[0]}
    shape=standardized_marginal_shapes(bank['m'],bank['validity'])
    raw=[]
    for data in [source_data,target_data]:
        with np.load(data/'raw-references.npz',allow_pickle=False) as a:raw.append({k:a[k] for k in a.files if k!='row_id'})
    raw={k:np.concatenate([a[k] for a in raw]) for k in raw[0]}
    assert file_hash(edge_bank)==json.loads(edge_bank.with_suffix('.json').read_text())['artifact_sha256']
    with np.load(edge_bank,allow_pickle=False) as a:
        np.testing.assert_array_equal(a['row_id'],[r['row_id'] for r in tr]);edges,valid,lengths=a['edges'],a['validity'],a['lengths']
    masters=np.load(target_data/'masters.npy',mmap_mode='r');torch.set_num_threads(2)
    directories=[('',source_root/'statistical-corrected'),('',source_root/'raw-corrected'),
        ('neural-aligned',source_root/'neural-aligned'),('neural-pair',source_root/'neural-pair'),
        ('learned-pooling',source_root/'learned-pooling')]
    selected={'z-pls','z-pca','m-pls','shape-pls','m+z-pls','shape+z-pls',
              'raw:agreement-pls','raw:marginal+covariance+cumulant+window-pls'}
    output.mkdir(parents=True,exist_ok=True);checks=[]
    for alias,directory in directories:
        for p in sorted(directory.rglob('*.json')):
            info=json.loads(p.read_text())
            if 'identity' not in info:continue
            old=info['identity'];name=alias or old['method']
            if not alias and name not in selected:continue
            assert old['manifest_sha256']==file_hash(source_data/'manifest.json')
            assert info['predictions_sha256']==file_hash(p.with_suffix('.npz'))
            with np.load(p.with_suffix('.npz'),allow_pickle=False) as a:
                train=a['train_indices'];old_eval=a['evaluation_indices'];old_pred=a['prediction']
            assert all(sr[i]['role']=='training_pool' for i in train)
            assert not {sr[i]['master_id'] for i in train}&{r['master_id'] for r in tr}
            ident=dict(method=name,seed=old['seed'],manifest_sha256=file_hash(target_data/'manifest.json'),
                source_manifest_sha256=file_hash(source_data/'manifest.json'),source_fit_sha256=file_hash(p),
                config_sha256=file_hash(config),target_bank_sha256=file_hash(target_bank),target_edge_bank_sha256=file_hash(edge_bank),
                evaluator_sha256=file_hash(Path(__file__)),model_code_sha256={f:file_hash(Path(f)) for f in
                    ['src/spi_edge_pool.py','src/representation_state_neural.py','src/interaction_share_learning.py','src/representation_screen.py']})
            stem=output/f"{name.replace('+','_').replace(':','_')}-n{info['labels_total']}-s{old['seed']}"
            if stem.with_suffix('.json').exists():
                saved=json.loads(stem.with_suffix('.json').read_text());assert saved['identity']==ident
                assert saved['predictions_sha256']==file_hash(stem.with_suffix('.npz'));continue
            replay=None
            if alias:
                assert info['details']['checkpoint_sha256']==file_hash(p.with_suffix('.pt'))
                ck=torch.load(p.with_suffix('.pt'),map_location='cpu',weights_only=False)
                model=make_encoder(ck['spec']);model.load_state_dict(ck['state_dict']);model.eval();prediction=np.empty(len(tr))
                for m,t in sorted({(r['M'],r['T']) for r in tr}):
                    idx=np.array([i for i,r in enumerate(tr) if (r['M'],r['T'])==(m,t)])
                    for start in range(0,len(idx),16):
                        ii=idx[start:start+16]
                        if alias=='learned-pooling':
                            assert np.all(lengths[ii]==m*(m-1));x=pack_inputs(edges[ii,:m*(m-1)],valid[ii])
                        else:x=np.stack([observed_view(masters[tr[i]['master_index']],m,t) for i in ii])
                        prediction[ii]=predict(model,torch.tensor(x,dtype=torch.float32),16)
            else:
                view,head=name.rsplit('-',1)
                if view.startswith('raw:'):
                    keys=view[4:].split('+');active={'u':np.concatenate([raw[k] for k in keys],axis=1)};view='u'
                else:
                    active=bank
                    if 'shape' in view:active={**bank,'m':shape};view=view.replace('shape','m')
                transform,model=fit_statistical(active,view,train,y,cfg['methods']['preprocessing'],head,*info['details']['chosen'])
                replay=np.clip(model.predict(transform.transform(active,old_eval)).reshape(-1),0,1)
                np.testing.assert_allclose(replay,old_pred,atol=1e-9,rtol=0)
                replay=float(np.max(abs(replay-old_pred)))
                prediction=np.clip(model.predict(transform.transform(active,np.arange(n,n+len(tr)))).reshape(-1),0,1)
            _atomic_savez(stem.with_suffix('.npz'),dict(prediction=prediction,target=np.array([r['target'] for r in tr]),
                train_indices=train,evaluation_indices=np.arange(len(tr)),row_id=np.array([r['row_id'] for r in tr])))
            _atomic_json(stem.with_suffix('.json'),dict(identity=ident,labels_total=info['labels_total'],
                predictions_sha256=file_hash(stem.with_suffix('.npz')),source_fit=str(p),source_replay_max_difference=replay,
                details={'no_target_fitting':True,'source_chosen_settings':info['details'].get('chosen')},status='prospective_regime_transfer'))
            checks.append(dict(method=name,labels=info['labels_total'],seed=old['seed'],source_replay_max_difference=replay))
            print(f'[DONE] {stem.name}',flush=True)
    expected=11*9
    paths=list(output.glob('*.json'));assert len(paths)==expected,(len(paths),expected)
    print(dict(complete_fits=len(paths),new_fits=len(checks)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','source-data','source-root','target-data','target-bank','edge-bank','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.config,a.source_data,a.source_root,a.target_data,a.target_bank,a.edge_bank,a.output)
