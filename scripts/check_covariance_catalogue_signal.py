"""Post-result attribution: nonlinear MI/dcorr values versus their agreements.

Fixed five-SPI panel excludes Gaussian MI, which is a function of correlation.
All means, rich summaries, normalized shapes and ten z coordinates use identical
label budgets and the existing PLS tuning. This is not fresh confirmation.
"""
import json
from pathlib import Path
import numpy as np
import yaml
from src.interaction_share_learning import fit_statistical,select_statistical,standardized_marginal_shapes
from src.representation_screen import training_subsets
from src.representation_state_data import source_pool_for_seed,file_hash


def main():
    root=Path('results/covariance_modulation_260909');data=Path('data/covariance_modulation_260909')
    output=root/'catalogue-signal';output.mkdir(exist_ok=True)
    config=Path('configs/analysis/covariance-modulation-260909.yaml');cfg=yaml.safe_load(config.read_text())
    rows=json.loads((data/'manifest.json').read_text())['rows']
    names=['dcorr','dcorr_biased','mi_kraskov_NN-4','mi_kraskov_NN-4_DCE-AUTO','mi_kernel_W-0.25']
    path=root/'gadi-analysis/features.npz'
    with np.load(path,allow_pickle=False) as a:
        order=a['spi_order'].tolist();idx=[order.index(x) for x in names]
        rich=a['X_m'].reshape(len(rows),len(order),23)[:,idx,:].reshape(len(rows),-1)
        means=rich.reshape(len(rows),len(idx),23)[:,:,0]
        valid=a['X_validity'][:,idx]
        left,right=np.triu_indices(len(order),1);keep=np.isin(left,idx)&np.isin(right,idx)
        z=a['X_z'][:,keep];assert z.shape[1]==10
    panels={'nlin-means-pls':{'m':means},'nlin-m-pls':{'m':rich},
            'nlin-shape-pls':{'m':standardized_marginal_shapes(rich,valid)},'nlin-z-pls':{'z':z}}
    y=np.array([r['target'] for r in rows]);strata=np.array([r['coupling_index'] for r in rows])
    evaluation=np.array([i for i,r in enumerate(rows) if r['role']=='evaluation'])
    for family in cfg['generator']['families']:
        pool=np.array([i for i,r in enumerate(rows) if r['role']=='training_pool' and r['family']==family])
        for seed in cfg['methods']['subset_seeds']:
            cohort=source_pool_for_seed(rows,pool,cfg,seed)
            for n,train in training_subsets(strata,cohort,cfg['sampling']['labelled_training_masters_per_coupling'],seed).items():
                for name,bank in panels.items():
                    stem=output/family/f'{name}-n{n}-s{seed}';stem.parent.mkdir(exist_ok=True)
                    if stem.with_suffix('.json').exists():raise FileExistsError(stem)
                    view=next(iter(bank));chosen,details=select_statistical(bank,view,train,y,strata,cfg['methods'],seed,'pls')
                    transform,model=fit_statistical(bank,view,train,y,cfg['methods']['preprocessing'],'pls',*chosen)
                    prediction=np.clip(model.predict(transform.transform(bank,evaluation)).reshape(-1),0,1)
                    np.savez_compressed(stem.with_suffix('.npz'),prediction=prediction,target=y[evaluation],
                        train_indices=train,evaluation_indices=evaluation,row_id=np.array([rows[i]['row_id'] for i in evaluation]))
                    details['chosen']=chosen
                    ident=dict(method=name,source_family=family,seed=seed,manifest_sha256=file_hash(data/'manifest.json'),
                               feature_bank_sha256=file_hash(path),panel=names,script_sha256=file_hash(Path(__file__)))
                    stem.with_suffix('.json').write_text(json.dumps(dict(identity=ident,details=details,labels_total=len(train),
                       predictions_sha256=file_hash(stem.with_suffix('.npz')),status='post_result_attribution_no_fresh_confirmation'),indent=2)+'\n')
    print('Completed72matched attribution fits')


if __name__=='__main__':main()
