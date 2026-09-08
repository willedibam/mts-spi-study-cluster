"""Matched small statistical heads for the covariance-modulation pilot."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import yaml
from sklearn.isotonic import IsotonicRegression
from src.interaction_share_learning import fit_statistical, select_statistical, standardized_marginal_shapes
from src.representation_screen import training_subsets
from src.representation_state_data import file_hash,load_state_data,source_pool_for_seed
from src.run_external_corpus import _atomic_json,_atomic_savez


def run(config,data,output,selected,feature_bank=None):
    cfg=yaml.safe_load(config.read_text());manifest,_=load_state_data(data,config)
    rows=manifest['rows'];target=np.asarray([r['target'] for r in rows]);strata=np.asarray([r['coupling_index'] for r in rows])
    evaluation=np.asarray([i for i,r in enumerate(rows) if r['role']=='evaluation'])
    with np.load(data/'raw-references.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['row_id'],[r['row_id'] for r in rows])
        raw={k:a[k] for k in a.files if k!='row_id'}
    bank={}
    if feature_bank:
        with np.load(feature_bank,allow_pickle=False) as a:
            np.testing.assert_array_equal(a['row_id'],[r['row_id'] for r in rows])
            assert a['manifest_sha256'].item()==file_hash(data/'manifest.json')
            bank={k:a['X_'+k] for k in ['m','g','z','validity']}
        shapes=standardized_marginal_shapes(bank['m'],bank['validity'])
    identity=dict(protocol_sha256=file_hash(config),manifest_sha256=file_hash(data/'manifest.json'),
                  feature_bank_sha256=file_hash(feature_bank) if feature_bank else None,
                  code_sha256={p:file_hash(Path(p)) for p in [__file__,'src/interaction_share_learning.py','src/representation_screen.py']})
    for family in cfg['generator']['families']:
        pool=np.asarray([i for i,r in enumerate(rows) if r['role']=='training_pool' and r['family']==family])
        for seed in cfg['methods']['subset_seeds']:
            cohort=source_pool_for_seed(rows,pool,cfg,seed)
            for n,train in training_subsets(strata,cohort,cfg['sampling']['labelled_training_masters_per_coupling'],seed).items():
                for name in selected:
                    stem=output/family/f'{name.replace("+","_")}-n{n}-s{seed}'
                    ident=dict(identity,method=name,source_family=family,n_per_coupling=n,seed=seed)
                    if stem.with_suffix('.json').exists():
                        old=json.loads(stem.with_suffix('.json').read_text())
                        assert old['identity']==ident and old['predictions_sha256']==file_hash(stem.with_suffix('.npz'))
                        continue
                    start=time.perf_counter();details={}
                    if name=='median': pred=np.full(len(evaluation),np.median(target[train]))
                    elif name=='moment-calibrated':
                        model=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(raw['moment_proxy'][train,0],target[train])
                        pred=model.predict(raw['moment_proxy'][evaluation,0])
                    else:
                        view,head=name.rsplit('-',1)
                        if view.startswith('raw:'):
                            keys=view[4:].split('+'); active={'u':np.concatenate([raw[k] for k in keys],axis=1)};view='u'
                        else:
                            active=bank
                            if 'shape' in view:active={**bank,'m':shapes};view=view.replace('shape','m')
                        chosen,details=select_statistical(active,view,train,target,strata,cfg['methods'],seed,head)
                        transform,model=fit_statistical(active,view,train,target,cfg['methods']['preprocessing'],head,*chosen)
                        pred=model.predict(transform.transform(active,evaluation)).reshape(-1);details['chosen']=chosen
                    _atomic_savez(stem.with_suffix('.npz'),dict(prediction=np.clip(pred,0,1),target=target[evaluation],
                                      train_indices=train,evaluation_indices=evaluation,row_id=np.asarray([rows[i]['row_id'] for i in evaluation])))
                    _atomic_json(stem.with_suffix('.json'),dict(identity=ident,details=details,labels_total=len(train),
                                 seconds=time.perf_counter()-start,predictions_sha256=file_hash(stem.with_suffix('.npz')),
                                 status='prospectively_specified_exploratory_pilot'))
                    print(f'[DONE] {family}/{stem.name}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','data','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--methods',nargs='+',required=True);p.add_argument('--feature-bank',type=Path)
    a=p.parse_args();run(a.config,a.data,a.output,a.methods,a.feature_bank)
