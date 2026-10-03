"""Development-gated band-swap comparison; means primary, no held-label tuning."""
import argparse,json,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from scripts.build_band_swap import DATA,OUT
from scripts.spi_baseline_exploration import project_features,sha
from scripts.analyze_native_coupling import extract as extract_common
from src.spi_spi_contract import build_unified_feature_values
from src.corpus_geometry import fit_geometry_transform
from src.utils import slugify


CORPUS='band-swap-261004'


def configure(run):
    global DATA,OUT,CORPUS
    from scripts.spi_baseline_exploration import ROOT
    CORPUS=run
    DATA=ROOT/'data/representation'/run
    OUT=ROOT/'results/representation'/run


def get_rows(stage):
    records=json.loads((DATA/'manifest.json').read_text())['rows']
    return pd.DataFrame(records[:64] if stage=='development' else records)


def extract(stage):
    target=OUT/stage;target.mkdir(parents=True,exist_ok=True)
    extract_common(DATA,target,corpus=CORPUS,row_limit=64 if stage=='development' else None)
    rows=get_rows(stage);rng=np.random.default_rng(261004);permuted=[]
    for _,r in rows.iterrows():
        folder=DATA/'mpis'/CORPUS/f"{r.corpus_index+1:04d}-{slugify(r.row_id,'dataset')}"
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


def diagnostic_features(bank,rows,train,target):
    """Rank on training data only; validation AUCs explain, never select, readouts."""
    with np.load(target/'features.npz') as archive:names=archive['spi_order']
    pair_names=np.array([f'{names[a]} | {names[b]}' for a,b in zip(*np.triu_indices(len(names),1))])
    y=rows.label.eq(sorted(rows.label.unique())[-1]).to_numpy();records=[]
    for method,labels in [('mean',names),('z',pair_names)]:
        transform=fit_geometry_transform(bank[method][train],scaling='standard',minimum_valid_fraction=.95)
        x=np.clip(transform.transform(bank[method]),-5,5)
        effect=x[train & y].mean(axis=0)-x[train & ~y].mean(axis=0)
        order=np.argsort(-np.abs(effect))[:10]
        for rank,j in enumerate(order,1):
            records.append(dict(method=method,training_rank=rank,feature=labels[transform.keep_indices[j]],
                training_standardized_difference=effect[j],evaluation_AUC=roc_auc_score(y[~train],x[~train,j]*np.sign(effect[j]))))
    pd.DataFrame(records).to_csv(target/'diagnostic-features.csv',index=False)


def analyze(stage,direct_only=False):
    target=OUT/stage;target.mkdir(parents=True,exist_ok=True);rows=get_rows(stage)
    train=rows.block.lt(24 if stage=='development' else 32).to_numpy();test=~train
    bank={}
    with np.load(DATA/'direct-features.npz') as a:
        np.testing.assert_array_equal(a['row_id'][:len(rows)],rows.row_id)
        for key in a.files:
            if key!='row_id':bank[key]=a[key][:len(rows)]
    if 'band_marginals' in bank:bank['band_mean']=bank['band_marginals'][:,::7]
    if not direct_only:
        with np.load(target/'features.npz') as a:
            np.testing.assert_array_equal(a['row_id'],rows.row_id)
            for key in ['mean','distribution','z','z_validity']:bank[key]=a[key]
        with np.load(target/'ablation.npz') as a:bank['z_shuffled']=a['z_shuffled']
        diagnostic_features(bank,rows,train,target)
        # Secondary robustness check: do not let training missingness define
        # these coordinates. Primary z retains the frozen 95% rule.
        bank['z_complete']=bank['z']
        bank['z_shuffled_complete']=bank['z_shuffled']
        for base in ['z','z_shuffled']:
            bank[base+'_center']=bank[base]
            bank[base+'_center_complete']=bank[base]
    projections={};readouts={};diagnostics={}
    for name,x in bank.items():
        try:
            projection,d,h=project_features(x[train],x[test],dimensions=20,
                standard='_center' not in name,valid=1.0 if name.endswith('_complete') else .95)
        except RuntimeError as error:
            if str(error)!='no features pass the variance gate':raise
            # A constant validity mask contains no training information. Its
            # control remains an intercept-only readout, without invented noise.
            d,h=np.zeros((train.sum(),1)),np.zeros((test.sum(),1))
            projections[name]=(d,h);readouts[name]=(d,h,'logistic')
            diagnostics[name]=dict(selected_features=0,components=0,explained_variance=None,constant_control=True)
            continue
        projections[name]=(d,h);readouts[name]=(d,h,'logistic')
        diagnostics[name]=dict(selected_features=len(projection.transform.keep_indices),components=d.shape[1],
            explained_variance=float(projection.pca.explained_variance_ratio_.sum()),
            evaluation_selected_missing_fraction=float(np.mean(~np.isfinite(x[test][:,projection.transform.keep_indices]))))
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
    reference=('band_z' if 'band_z' in bank else 'probe_z') if direct_only else 'z';paired=[]
    for comparator in pivot.columns.drop(reference):
        delta=(pivot[reference]-pivot[comparator]).groupby('block').mean().to_numpy()
        rng=np.random.default_rng(261003)
        boot=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(axis=1)
        low,high=np.quantile(boot,[.025,.975])
        paired.append(dict(comparison=reference+' - '+comparator,difference=delta.mean(),low=low,high=high))
    for variant in ['z_complete','z_center','z_center_complete']:
        if variant in pivot:
            for comparator in ['mean','mean_full','mean_RBF','mean_trees','distribution',variant.replace('z','z_shuffled',1)]:
                delta=(pivot[variant]-pivot[comparator]).groupby('block').mean().to_numpy()
                rng=np.random.default_rng(261003)
                boot=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(axis=1)
                low,high=np.quantile(boot,[.025,.975])
                paired.append(dict(comparison=variant+' - '+comparator,difference=delta.mean(),low=low,high=high))
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
            centered=dict(gate,z_detectable=bool(ba['z_center']>=.80))
            provenance['centered_development_goal']=dict(**centered,all_pass=all(centered.values()),
                qualification='Center-only z refinement selected after initial development result; requires untouched held confirmation')
    (target/(prefix+'analysis.json')).write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.DataFrame(metrics).round(4).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','analyze','direct']);p.add_argument('--stage',choices=['development','final'],default='development');p.add_argument('--run',default='band-swap-261004');a=p.parse_args()
    configure(a.run)
    with threadpool_limits(limits=4):
        if a.action=='extract':extract(a.stage)
        else:analyze(a.stage,direct_only=a.action=='direct')
