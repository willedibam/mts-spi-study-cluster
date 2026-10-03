"""Development-gated band-swap comparison; means primary, no held-label tuning."""
import argparse,json,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from scripts.build_band_swap import DATA,OUT
from scripts.spi_baseline_exploration import project_features,sha
from scripts.analyze_native_coupling import extract as extract_common
from src.spi_spi_contract import build_unified_feature_values
from src.corpus_geometry import fit_geometry_transform
from src.utils import slugify


def get_rows(stage):
    records=json.loads((DATA/'manifest.json').read_text())['rows']
    return pd.DataFrame(records[:64] if stage=='development' else records)


def extract(stage):
    target=OUT/stage;target.mkdir(parents=True,exist_ok=True)
    extract_common(DATA,target,corpus='band-swap-261004',row_limit=64 if stage=='development' else None)
    rows=get_rows(stage);rng=np.random.default_rng(261004);permuted=[]
    for _,r in rows.iterrows():
        folder=DATA/'mpis/band-swap-261004'/f"{r.corpus_index+1:04d}-{slugify(r.row_id,'dataset')}"
        meta=json.loads((folder/'meta.json').read_text());names=[v['name'] for v in meta['pyspi']['spis']]
        with np.load(folder/'spi_mpis.npz') as a:
            mask=~np.eye(r.M,dtype=bool);mpis={}
            for name in names:
                x=a[name].copy();values=x[mask].copy();rng.shuffle(values);x[mask]=values
                np.testing.assert_array_equal(np.sort(values),np.sort(a[name][mask]))
                mpis[name]=x
        z,_,_=build_unified_feature_values(mpis,names);permuted.append(z)
    np.savez_compressed(target/'ablation.npz',z_shuffled=np.array(permuted),row_id=rows.row_id.to_numpy())


def fitted_logistic(x,y):
    for cap in (3000,20000):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            model=LogisticRegression(C=1,max_iter=cap).fit(x,y)
        if not any(issubclass(w.category,ConvergenceWarning) for w in caught):return model
    raise RuntimeError('Logistic fit did not converge')


def analyze(stage,direct_only=False):
    target=OUT/stage;target.mkdir(parents=True,exist_ok=True);rows=get_rows(stage)
    train=rows.block.lt(24 if stage=='development' else 32).to_numpy();test=~train
    bank={}
    with np.load(DATA/'direct-features.npz') as a:
        np.testing.assert_array_equal(a['row_id'][:len(rows)],rows.row_id)
        for key in ['band_z','band_marginals','raw_covariance','raw_Pearson','spectra']:bank[key]=a[key][:len(rows)]
    bank['band_mean']=bank['band_marginals'][:,::7]
    if not direct_only:
        with np.load(target/'features.npz') as a:
            np.testing.assert_array_equal(a['row_id'],rows.row_id)
            for key in ['mean','distribution','z','z_validity']:bank[key]=a[key]
        with np.load(target/'ablation.npz') as a:bank['z_shuffled']=a['z_shuffled']
    projections={};readouts={};diagnostics={}
    for name,x in bank.items():
        projection,d,h=project_features(x[train],x[test],dimensions=20)
        projections[name]=(d,h);readouts[name]=(d,h,'logistic')
        diagnostics[name]=dict(selected_features=len(projection.transform.keep_indices),components=d.shape[1],
            explained_variance=float(projection.pca.explained_variance_ratio_.sum()))
    if 'mean' in bank:
        transform=fit_geometry_transform(bank['mean'][train],scaling='standard',minimum_valid_fraction=.95)
        x=np.clip(transform.transform(bank['mean']),-5,5)
        for name,kind in [('mean_full','logistic'),('mean_RBF','rbf'),('mean_trees','trees')]:readouts[name]=(x[train],x[test],kind)
    metrics=[];predictions=[]
    for method,(d,h,kind) in readouts.items():
        labels=rows.loc[train,'label'].to_numpy()
        if kind=='logistic':model=fitted_logistic(d,labels)
        elif kind=='rbf':model=SVC(C=1,gamma='scale').fit(d,labels)
        else:model=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,random_state=261003,n_jobs=4).fit(d,labels)
        f=rows.loc[test,['row_id','label','block']].copy();f['method']=method;f['predicted']=model.predict(h);f['correct']=f.label==f.predicted
        predictions.append(f);g=f.groupby('block').correct.mean().to_numpy();rng=np.random.default_rng(261003)
        boot=g[rng.integers(len(g),size=(5000,len(g)))].mean(axis=1);low,high=np.quantile(boot,[.025,.975])
        metrics.append(dict(method=method,n=len(f),BA=balanced_accuracy_score(f.label,f.predicted),low=low,high=high,chance=.5))
    prefix='direct-' if direct_only else ''
    scores=pd.DataFrame(metrics)
    scores.to_csv(target/(prefix+'metrics.csv'),index=False)
    predictions=pd.concat(predictions)
    predictions.to_csv(target/(prefix+'predictions.csv'),index=False)
    pivot=predictions.pivot(index=['row_id','block'],columns='method',values='correct').astype(float)
    reference='band_z' if direct_only else 'z';paired=[]
    for comparator in pivot.columns.drop(reference):
        delta=(pivot[reference]-pivot[comparator]).groupby('block').mean().to_numpy()
        rng=np.random.default_rng(261003)
        boot=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(axis=1)
        low,high=np.quantile(boot,[.025,.975])
        paired.append(dict(comparison=reference+' - '+comparator,difference=delta.mean(),low=low,high=high))
    pd.DataFrame(paired).to_csv(target/(prefix+'paired.csv'),index=False)
    np.savez_compressed(target/(prefix+'projections.npz'),**{k+s:v for k,pair in projections.items() for s,v in zip(('_train','_test'),pair)})
    provenance=dict(stage=stage,rows=rows.row_id.tolist(),diagnostics=diagnostics,
        code_sha256=sha(Path(__file__)),direct_features_sha256=sha(DATA/'direct-features.npz'),
        uncertainty='Conditional paired block bootstrap; fitted models fixed; no equivalence claim')
    if not direct_only:
        provenance['features_sha256']=sha(target/'features.npz')
        provenance['ablation_sha256']=sha(target/'ablation.npz')
        if stage=='development':
            ba=scores.set_index('method').BA
            gate=dict(mean_readouts_weak=bool(ba[['mean','mean_full','mean_RBF','mean_trees']].max()<=.65),
                z_detectable=bool(ba['z']>=.80),validity_weak=bool(ba['z_validity']<=.65))
            provenance['development_goal']=dict(**gate,all_pass=all(gate.values()),
                qualification='Exploratory small-validation gate, not a proof of marginal equality; inspect before held release')
    (target/(prefix+'analysis.json')).write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.DataFrame(metrics).round(4).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','analyze','direct']);p.add_argument('--stage',choices=['development','final'],default='development');a=p.parse_args()
    with threadpool_limits(limits=4):
        if a.action=='extract':extract(a.stage)
        else:analyze(a.stage,direct_only=a.action=='direct')
