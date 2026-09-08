"""Build audited p90 bank and run fixed grouped readouts for the96-view scout."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.model_selection import GroupKFold
from src.interaction_share_learning import fit_statistical,standardized_marginal_shapes
from src.mpi_representation_baselines import summarize_mpis
from src.representation_attribution import rich_marginals
from src.representation_state_data import file_hash
from src.run_external_corpus import _array_sha256
from src.spi_spi_contract import build_unified_features,schema_sha256
from src.utils import slugify


def build(data,output):
    path=output/'features.npz';manifest=json.loads((data/'manifest.json').read_text());rows=manifest['rows']
    if path.exists():
        assert file_hash(path)==json.loads(path.with_suffix('.json').read_text())['artifact_sha256']
        return path
    banks={k:[] for k in ['m','g','z','validity']};order=None;sources=[];versions=set()
    with np.load(data/'views.npz',allow_pickle=False) as raw:
        assert file_hash(data/'views.npz')==manifest['artifacts']['views.npz']
        for row in rows:
            directory=data/'mpis/oscillatory-coorganization-scout-260909'/f"{row['corpus_index']:04d}-{slugify(row['row_id'],'dataset')}"
            meta=json.loads((directory/'meta.json').read_text());mpi=directory/'spi_mpis.npz'
            assert meta['status']=='complete' and meta['normalise'] is False
            assert meta['dataset_name']==row['row_id'] and (meta['M'],meta['T'])==(row['M'],row['T'])
            assert meta['source']['archive_sha256']==manifest['artifacts']['views.npz']
            assert meta['source']['member_sha256']==_array_sha256(raw[row['row_id']])
            assert meta['pyspi']['config_sha256']==file_hash(Path('configs/pyspi/benchmarked_p90.yaml'))
            names=[s['name'] for s in meta['pyspi']['spis']]
            if order is None:order=names
            assert names==order and len(order)==289
            with np.load(mpi,allow_pickle=False) as a:
                assert a.files==order;mpis={k:a[k] for k in order}
            _,graph,valid=summarize_mpis(mpis,order);features=build_unified_features(mpis,order,metric='pearson')
            for k,v in zip(banks,[rich_marginals(mpis,order),graph,features.z,valid],strict=True):banks[k].append(v)
            sources.append(dict(row_id=row['row_id'],mpi_sha256=file_hash(mpi),meta_sha256=file_hash(directory/'meta.json'),
                                compute_seconds=meta['job']['compute_seconds'],valid_spis=int(valid.sum())))
            versions.add(json.dumps(meta['pyspi']['version'],sort_keys=True))
    assert len(versions)==1
    np.savez_compressed(path,**{'X_'+k:np.asarray(v) for k,v in banks.items()},spi_order=np.asarray(order),
                        row_id=np.asarray([r['row_id'] for r in rows]),schema_sha256=schema_sha256(features.schema),
                        manifest_sha256=file_hash(data/'manifest.json'))
    path.with_suffix('.json').write_text(json.dumps(dict(artifact_sha256=file_hash(path),sources=sources,
        versions=list(versions),builder_sha256=file_hash(Path(__file__))),indent=2)+'\n')
    return path


def analyze(data,output):
    output.mkdir(parents=True,exist_ok=True);path=build(data,output)
    rows=json.loads((data/'manifest.json').read_text())['rows'];y=np.array([r['target'] for r in rows])
    groups=np.array([r['block'] for r in rows]);sizes=np.array([r['M'] for r in rows])
    with np.load(path,allow_pickle=False) as a:bank={k:a['X_'+k] for k in ['m','g','z','validity']}
    shapes=standardized_marginal_shapes(bank['m'],bank['validity'])
    with np.load(data/'raw.npz',allow_pickle=False) as a:
        np.testing.assert_array_equal(a['row_id'],[r['row_id'] for r in rows]);raw={k:a[k] for k in a.files if k!='row_id'}
    models={
        'm-pls':(bank,'m','pls'),'z-pls':(bank,'z','pls'),'z-pca':(bank,'z','pca'),
        'm+z-pls':(bank,'m+z','pls'),'shape-pls':({**bank,'m':shapes},'m','pls'),
        'shape+z-pls':({**bank,'m':shapes},'m+z','pls'),'g-pls':(bank,'g','pls'),
        'validity-pls':(bank,'validity','pls'),
        'raw-spectrum-pls':({'u':raw['spectrum']},'u','pls'),
        'raw-moments-pls':({'u':np.concatenate([raw[k] for k in ['raw_marginal','raw_covariance','raw_cumulant','raw_window']],axis=1)},'u','pls'),
        'phase-envelope-marginals-pls':({'u':np.c_[raw['phase_summary'],raw['envelope_summary']]},'u','pls'),
        'direct-agreement-pls':({'u':raw['direct_agreement']},'u','pls')}
    methods=yaml.safe_load(Path('configs/analysis/covariance-modulation-260909.yaml').read_text())['methods']
    order=np.random.default_rng(260909137).permutation(24);test_blocks=order[12:]
    evaluation=np.flatnonzero(np.isin(groups,test_blocks));records=[]
    splits={};fits=output/'fits';fits.mkdir(exist_ok=True)
    for count in [4,8,12]:
        train=np.flatnonzero(np.isin(groups,order[:count])&(sizes==16))
        folds=[(train[a],train[b]) for a,b in GroupKFold(2).split(train,y[train],groups[train])]
        assert not set(groups[train])&set(groups[evaluation])
        for a,b in folds:assert not set(groups[a])&set(groups[b])
        splits[count]=dict(train=train.tolist(),evaluation=evaluation.tolist(),folds=[dict(fit=a.tolist(),validation=b.tolist()) for a,b in folds])
        for name,(active,view,head) in models.items():
            stem=fits/f'{name.replace("+","_")}-b{count}'
            if stem.with_suffix('.json').exists():raise FileExistsError(stem)
            candidates=([(k,None) for k in [1,2,4]] if head=='pls' else
                        [(k,a) for k in [1,2,4,8,16] for a in [100,10,1,.1,.01]])
            scores=[]
            for k,alpha in candidates:
                losses=[]
                for fit,val in folds:
                    transform,model=fit_statistical(active,view,fit,y,methods['preprocessing'],head,k,alpha)
                    pred=np.clip(model.predict(transform.transform(active,val)).reshape(-1),0,1)
                    losses.append(float(abs(pred-y[val]).mean()))
                scores.append(float(np.mean(losses)))
            selected=int(np.argmin(scores));k,alpha=candidates[selected]
            transform,model=fit_statistical(active,view,train,y,methods['preprocessing'],head,k,alpha)
            pred=np.clip(model.predict(transform.transform(active,evaluation)).reshape(-1),0,1)
            np.savez_compressed(stem.with_suffix('.npz'),prediction=pred,target=y[evaluation],train_indices=train,evaluation_indices=evaluation)
            stem.with_suffix('.json').write_text(json.dumps(dict(method=name,labels=len(train),chosen=candidates[selected],
                candidates=candidates,CV_MAE=scores,predictions_sha256=file_hash(stem.with_suffix('.npz'))),indent=2)+'\n')
            for m in [16,8]:
                mask=sizes[evaluation]==m;truth=y[evaluation][mask];prediction=pred[mask]
                records.append(dict(method=name,labels=len(train),training_blocks=count,M=m,
                    AUROC=float(roc_auc_score(truth,prediction)),balanced_accuracy=float(balanced_accuracy_score(truth,prediction>=.5)),
                    MAE=float(abs(truth-prediction).mean())))
    pd.DataFrame(records).to_csv(output/'summary.csv',index=False)
    (output/'splits.json').write_text(json.dumps(dict(permuted_blocks=order.tolist(),splits=splits,script_sha256=file_hash(Path(__file__)),
        feature_bank_sha256=file_hash(path),status='exploratory_scout_raw_feasibility_already_inspected_no_confirmation'),indent=2)+'\n')
    print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.data,a.output)
