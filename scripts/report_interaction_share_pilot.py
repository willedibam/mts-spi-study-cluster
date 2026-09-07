"""Report family/observation cells and paired master-level uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from src.representation_state_data import file_hash, load_state_data
from src.representation_screen import bootstrap_group_means
from src.run_external_corpus import _atomic_json


def report(config,data,inputs,output):
    p=yaml.safe_load(config.read_text()); manifest,_=load_state_data(data,config)
    rows=manifest['rows']; families=p['generator']['families']; seeds=p['methods']['subset_seeds']
    budgets=p['sampling']['labelled_training_masters_per_coupling']
    fits={}; provenance=[]
    for directory in inputs:
        for path in sorted(directory.rglob('*.json')):
            record=json.loads(path.read_text())
            if 'identity' not in record or 'predictions_sha256' not in record: continue
            ident=record['identity']
            assert ident['protocol_sha256']==file_hash(config)
            assert ident['manifest_sha256']==file_hash(data/'manifest.json')
            assert record['predictions_sha256']==file_hash(path.with_suffix('.npz'))
            key=(ident['method'],ident['source_family'],ident['n_per_coupling'],ident['seed'])
            if key in fits: raise ValueError(f'duplicate fit {key}')
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as a:
                ix=a['evaluation_indices']; train=a['train_indices']
                assert all(rows[i]['role']=='training_pool' and rows[i]['family']==key[1] for i in train)
                assert len(train)==key[2]*5
                assert not {rows[i]['master_id'] for i in train}&{rows[i]['master_id'] for i in ix}
                np.testing.assert_array_equal(a['row_id'],[rows[i]['row_id'] for i in ix])
                np.testing.assert_allclose(a['target'],[rows[i]['target'] for i in ix],rtol=0,atol=0)
                fits[key]=(ix,abs(a['prediction']-a['target']))
            provenance.append(dict(path=str(path),sha256=file_hash(path)))
    # Average errors across technical surrogate seeds, never ensemble their
    # predictions or count them as additional independent realizations.
    for head in ['pca','pls']:
        for family in families:
            for n in budgets:
                for seed in seeds:
                    keys=[(f'null{s}_z-{head}',family,n,seed) for s in [503,509,521]]
                    if all(key in fits for key in keys):
                        indices=fits[keys[0]][0]
                        assert all(np.array_equal(indices,fits[key][0]) for key in keys)
                        fits[(f'null_mean-{head}',family,n,seed)]=(indices,np.mean([fits[key][1] for key in keys],axis=0))
    names=sorted({k[0] for k in fits}); expected=len(families)*len(budgets)*len(seeds)
    complete=[name for name in names if sum(k[0]==name for k in fits)==expected]
    incomplete={name:sum(k[0]==name for k in fits) for name in names if name not in complete}
    if not complete: raise ValueError('No complete method yet')
    cells={}; primary={}; primary_boot={}
    for name in complete:
        directional=[]; directional_boot=[]
        for family in families:
            for destination in families:
                for cell in ['source','shift']:
                    shape=p['observations'][cell]
                    values=[]
                    for n in budgets:
                        errors=[]
                        for seed in seeds:
                            ix,error=fits[(name,family,n,seed)]
                            mask=np.asarray([rows[i]['family']==destination and rows[i]['M']==shape['M'] and rows[i]['T']==shape['T'] for i in ix])
                            assert mask.sum()==100
                            errors.append(error[mask])
                        values.append(np.mean(errors,axis=0))
                    values=np.asarray(values)
                    strata=np.asarray([rows[i]['coupling_index'] for i in ix[mask]])
                    # One row per master within each cell; average five trained subsets first.
                    assert len({rows[i]['master_id'] for i in ix[mask]})==len(strata)
                    boot=bootstrap_group_means(values,strata,2000,1729+families.index(family))
                    key=f'{family}->{destination}/{cell}'
                    cells.setdefault(key,{})[name]=dict(MAE=values.mean(axis=1).tolist(),
                         conditional_95_CI=np.quantile(boot,[.025,.975],axis=-1).T.tolist())
                    if family!=destination and cell=='shift':
                        directional.append(values); directional_boot.append(boot)
        primary[name]=np.mean(directional,axis=0)
        primary_boot[name]=np.mean(directional_boot,axis=0)
    summary={name:dict(MAE=values.mean(axis=-1).tolist(),conditional_95_CI=np.quantile(primary_boot[name],[.025,.975],axis=-1).T.tolist()) for name,values in primary.items()}
    comparisons={}
    pairs=[('z-pca','m-pca'),('m+z-pca','m-pca'),('z-pls','m-pls'),('m+z-pls','m-pls'),('z-pls','z-pca'),('z-pca','linear'),('z-pls','linear'),('z-pca','neural'),('z-pls','random_encoder')]
    pairs += [(left+'-'+head,right+'-'+head) for head in ['pca','pls','rbf']
              for left,right in [('z','shape'),('shape+z','shape'),('shape','m')]]
    pairs += [(v+'-rbf',v+'-pls') for v in ['m','shape','z']]
    pairs += [('z-'+head,'null_mean-'+head) for head in ['pca','pls']]
    for left,right in pairs:
        if left in primary and right in primary:
            delta=primary[left]-primary[right]; boot=primary_boot[left]-primary_boot[right]
            comparisons[left+'_minus_'+right]=dict(MAE_difference=delta.mean(axis=-1).tolist(),conditional_95_CI=np.quantile(boot,[.025,.975],axis=-1).T.tolist())
    output.mkdir(exist_ok=True,parents=True)
    _atomic_json(output/'results.json',dict(status='exploratory_conditional_on_fitted_models',primary=summary,cells=cells,paired_primary=comparisons,
                 total_label_budgets=p['sampling']['total_label_budgets'],incomplete_methods=incomplete,fit_provenance=provenance,
                 report_code_sha256=file_hash(Path(__file__)),protocol_sha256=file_hash(config)))
    lines=['# Interaction-share pilot','', 'Joint shift: held-out family, M16/T1000 to M8/T500; equal weight to both directions.',
           'Fresh exploratory evaluation; five labelled subsets, no pretraining. Lower MAE is better.','',
           '| Method | 10 labels | 20 labels | 40 labels |','|---|---:|---:|---:|']
    for name,item in summary.items(): lines.append('| '+name+' | '+' | '.join(f'{x:.4f}' for x in item['MAE'])+' |')
    lines+=['','Per-direction and per-cell results and paired intervals are in results.json. Intervals resample independent evaluation masters within nominal-share strata, conditional on the fitted models; they are exploratory and unadjusted for multiple comparisons.',f'Incomplete methods (not ranked): {incomplete}','']
    (output/'report.md').write_text('\n'.join(lines)); print('\n'.join(lines))


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    for key in ['config','data','output']: a.add_argument('--'+key,type=Path,required=True)
    a.add_argument('--inputs',type=Path,nargs='+',required=True)
    v=a.parse_args();report(v.config,v.data,v.inputs,v.output)
