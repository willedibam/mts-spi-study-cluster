"""Matched source-only fits for both interaction families; resumable by fit."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import sklearn
import yaml
from sklearn.isotonic import IsotonicRegression

from src.interaction_share_learning import fit_statistical, select_statistical, select_reference, standardized_marginal_shapes
from src.representation_screen import training_subsets
from src.representation_state_data import file_hash, load_state_data, source_pool_for_seed
from src.run_external_corpus import _atomic_json, _atomic_savez


def run(config_path,data,output,selected,feature_bank=None,marginal_mode='raw',method_prefix=''):
    protocol=yaml.safe_load(config_path.read_text())
    manifest,_=load_state_data(data,config_path)
    rows=manifest['rows']; methods=protocol['methods']
    target=np.asarray([r['target'] for r in rows]); strata=np.asarray([r['coupling_index'] for r in rows])
    evaluation=np.asarray([i for i,r in enumerate(rows) if r['role']=='evaluation'])
    with np.load(data/'raw-controls.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['row_id'],[r['row_id'] for r in rows])
        bank={'u':a['X_u']}; memory=a['memory']; references=a['references']
    if feature_bank:
        with np.load(feature_bank,allow_pickle=False) as a:
            np.testing.assert_array_equal(a['row_id'],[r['row_id'] for r in rows])
            assert a['manifest_sha256'].item()==file_hash(data/'manifest.json')
            bank.update({k:a['X_'+k] for k in ['m','z','validity']})
    if marginal_mode == 'shape':
        if feature_bank is None or any(name not in [v+'-'+h for v in ['m','m+z'] for h in ['pca','pls','rbf']] for name in selected):
            raise ValueError('shape sensitivity requires a feature bank and marginal-containing views')
        bank['m'] = standardized_marginal_shapes(bank['m'],bank['validity'])
    identity=dict(protocol_sha256=file_hash(config_path),manifest_sha256=file_hash(data/'manifest.json'),
                  feature_bank_sha256=file_hash(feature_bank) if feature_bank else None,
                  numpy=np.__version__,sklearn=sklearn.__version__,
                  code_sha256={p:file_hash(Path(p)) for p in [__file__,'src/interaction_share_learning.py','src/representation_screen.py']})
    if marginal_mode == 'shape': identity['marginal_mode'] = 'post_result_affine_invariance_control'
    for family in protocol['generator']['families']:
        pool=np.asarray([i for i,r in enumerate(rows) if r['role']=='training_pool' and r['family']==family])
        assert len({rows[i]['master_id'] for i in pool})==len(pool)
        assert not {rows[i]['master_id'] for i in pool}&{rows[i]['master_id'] for i in evaluation}
        for seed in methods['subset_seeds']:
            cohort = source_pool_for_seed(rows,pool,protocol,seed)
            for n,train in training_subsets(strata,cohort,protocol['sampling']['labelled_training_masters_per_coupling'],seed).items():
                for name in selected:
                    reported = method_prefix + (name.replace('m','shape',1) if marginal_mode == 'shape' else name)
                    stem=output/family/f'{reported.replace("+","_")}-n{n}-s{seed}'
                    ident={**identity,'method':reported,'source_family':family,'n_per_coupling':n,'seed':seed}
                    if stem.with_suffix('.json').exists():
                        old=json.loads(stem.with_suffix('.json').read_text())
                        if old['identity']!=ident or old['predictions_sha256']!=file_hash(stem.with_suffix('.npz')):
                            raise ValueError(f'Resume mismatch {stem}')
                        continue
                    start=time.perf_counter(); details={}
                    if name=='median':
                        prediction=np.full(len(evaluation),np.median(target[train]))
                    elif name=='memory':
                        model=IsotonicRegression(increasing='auto',out_of_bounds='clip').fit(memory[train],target[train])
                        prediction=model.predict(memory[evaluation])
                    elif name in ['linear','nonlinear']:
                        width=len(methods['raw_ridge_fractions'])
                        values=references[:, :width] if name=='linear' else references[:,width:]
                        col,model,details=select_reference(values,train,target,strata,seed)
                        prediction=model.predict(values[evaluation,col])
                        details['ridge_fraction']=methods['raw_ridge_fractions'][col]
                    else:
                        view,head=name.rsplit('-',1)
                        chosen,details=select_statistical(bank,view,train,target,strata,methods,seed,head)
                        transform,model=fit_statistical(bank,view,train,target,methods['preprocessing'],head,*chosen)
                        prediction=model.predict(transform.transform(bank,evaluation)).reshape(-1)
                        details['chosen']=chosen
                    prediction=np.clip(prediction,0,1)
                    _atomic_savez(stem.with_suffix('.npz'),dict(prediction=prediction,target=target[evaluation],
                                  train_indices=train,evaluation_indices=evaluation,row_id=np.asarray([rows[i]['row_id'] for i in evaluation])))
                    _atomic_json(stem.with_suffix('.json'),dict(identity=ident,details=details,labels_total=len(train),
                                 seconds=time.perf_counter()-start,train_indices=train.tolist(),evaluation_indices=evaluation.tolist(),
                                 predictions_sha256=file_hash(stem.with_suffix('.npz')),status='exploratory_pilot'))
                    print(f'[DONE] {family}/{stem.name} {time.perf_counter()-start:.1f}s',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','data','output']: p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--feature-bank',type=Path)
    p.add_argument('--methods',nargs='+',required=True)
    p.add_argument('--marginal-mode',choices=['raw','shape'],default='raw')
    p.add_argument('--method-prefix',default='')
    a=p.parse_args(); run(a.config,a.data,a.output,a.methods,a.feature_bank,a.marginal_mode,a.method_prefix)
