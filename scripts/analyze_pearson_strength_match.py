"""Four-level baseline comparison on raw strength-normalized recordings."""
import argparse,json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from scripts.build_pearson_strength_match import DATA,OUT,RUN
from scripts.analyze_native_coupling import extract as extract_common
from scripts.analyze_band_swap import fitted_logistic
from scripts.spi_baseline_exploration import project_features,sha
from src.corpus_geometry import fit_geometry_transform
from src.spi_spi_contract import build_unified_feature_values
from src.utils import slugify


def rows(stage):
    records=json.loads((DATA/'manifest.json').read_text())['rows']
    return pd.DataFrame([r for r in records if r['role']=='development'] if stage=='development' else records)


def extract(stage):
    target=OUT/stage;target.mkdir(parents=True,exist_ok=True)
    frame=rows(stage)
    extract_common(DATA,target,corpus=RUN,row_limit=len(frame))
    shuffled=[];rng=np.random.default_rng(261007)
    for _,r in frame.iterrows():
        folder=DATA/'mpis'/RUN/f"{r.corpus_index+1:04d}-{slugify(r.row_id,'dataset')}"
        meta=json.loads((folder/'meta.json').read_text());names=[s['name'] for s in meta['pyspi']['spis']]
        with np.load(folder/'spi_mpis.npz') as archive:
            mask=~np.eye(r.M,dtype=bool);mpis={}
            for name in names:
                a=archive[name].copy();v=a[mask].copy();rng.shuffle(v);a[mask]=v
                np.testing.assert_array_equal(np.sort(v),np.sort(archive[name][mask]));mpis[name]=a
        z,_,_=build_unified_feature_values(mpis,names);shuffled.append(z)
    np.savez_compressed(target/'shuffled.npz',z=np.array(shuffled),row_id=frame.row_id.to_numpy())


def analyze(stage):
    target=OUT/stage;target.mkdir(parents=True,exist_ok=True)
    allrows=rows(stage);keep=allrows.panel.eq('matched').to_numpy();frame=allrows.loc[keep].reset_index(drop=True)
    train=(frame.development_part.eq('train') if stage=='development' else frame.role.eq('development')).to_numpy()
    y=frame.label.to_numpy();chance=1/len(np.unique(y));bank={};projections={};readouts={};diagnostics={}
    with np.load(target/'features.npz') as a:
        np.testing.assert_array_equal(a['row_id'],allrows.row_id)
        for key in ['mean','distribution','z','z_validity']:bank[key]=a[key][keep]
        cov_index=list(a['spi_order']).index('cov_EmpiricalCovariance')
    np.testing.assert_allclose(bank['mean'][:,cov_index],frame.mean_covariance,atol=1e-12)
    bank['b']=bank['mean'][:,cov_index,None]
    bank['pearson_two']=np.c_[bank['b'],frame.mean_abs_Pearson]
    bank['z_complete']=bank['z'];bank['z_standard']=bank['z'];bank['mean_complete']=bank['mean']
    with np.load(target/'shuffled.npz') as a:bank['z_shuffled_complete']=a['z'][keep]
    for name,x in bank.items():
        standard=name not in ['z','z_complete','z_shuffled_complete']
        try:
            projection,d,h=project_features(x[train],x[~train],standard=standard,dimensions=20,valid=1. if name.endswith('_complete') else .95)
            diagnostics[name]=dict(selected_features=len(projection.transform.keep_indices),components=d.shape[1],
                validation_selected_missing_fraction=float(np.mean(~np.isfinite(x[~train][:,projection.transform.keep_indices]))))
        except RuntimeError as e:
            if str(e)!='no features pass the variance gate':raise
            d,h=np.zeros((train.sum(),1)),np.zeros(((~train).sum(),1));diagnostics[name]=dict(selected_features=0,constant=True)
        projections[name]=(d,h);readouts[name]=(d,h,'linear')
    for key in ['mean','b','pearson_two']:
        transform=fit_geometry_transform(bank[key][train],scaling='standard',minimum_valid_fraction=.95)
        x=np.clip(transform.transform(bank[key]),-5,5)
        for suffix,kind in [('full','linear'),('RBF','rbf'),('trees','trees')]:readouts[key+'_'+suffix]=(x[train],x[~train],kind)
    metrics=[];predictions=[]
    for name,(d,h,kind) in readouts.items():
        if kind=='linear':model=fitted_logistic(d,y[train])
        elif kind=='rbf':model=SVC(C=1,gamma='scale').fit(d,y[train])
        else:model=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,random_state=261003,n_jobs=4).fit(d,y[train])
        f=frame.loc[~train,['row_id','label','block']].copy();f['method']=name;f['predicted']=model.predict(h);f['correct']=f.label==f.predicted
        predictions.append(f);g=f.groupby('block').correct.mean().to_numpy();rng=np.random.default_rng(261007)
        boot=g[rng.integers(len(g),size=(5000,len(g)))].mean(axis=1);low,high=np.quantile(boot,[.025,.975])
        metrics.append(dict(method=name,n=len(f),BA=balanced_accuracy_score(f.label,f.predicted),low=low,high=high,chance=chance))
    score=pd.DataFrame(metrics);score.to_csv(target/'metrics.csv',index=False)
    predictions=pd.concat(predictions);predictions.to_csv(target/'predictions.csv',index=False)
    pivot=predictions.pivot(index=['row_id','block'],columns='method',values='correct').astype(float);paired=[]
    for method in pivot.columns.drop('z_complete'):
        delta=(pivot.z_complete-pivot[method]).groupby('block').mean().to_numpy();rng=np.random.default_rng(261007)
        boot=delta[rng.integers(len(delta),size=(5000,len(delta)))].mean(axis=1);lo,hi=np.quantile(boot,[.025,.975])
        paired.append(dict(comparison='z_complete - '+method,difference=delta.mean(),low=lo,high=hi))
    pd.DataFrame(paired).to_csv(target/'paired.csv',index=False)
    np.savez_compressed(target/'projections.npz',**{k+s:v for k,p in projections.items() for s,v in zip(('_train','_test'),p)})
    ba=score.set_index('method').BA
    gate=dict(covariance_mean_weak=bool(ba[['b','b_full','b_RBF','b_trees']].max()<=.27),
        signed_absolute_weak=bool(ba[['pearson_two','pearson_two_full','pearson_two_RBF','pearson_two_trees']].max()<=.32),z_detectable=bool(ba.z_complete>=.8))
    (target/'analysis.json').write_text(json.dumps(dict(stage=stage,train_rows=frame.loc[train,'row_id'].tolist(),test_rows=frame.loc[~train,'row_id'].tolist(),
        diagnostics=diagnostics,development_gate=dict(**gate,all_pass=all(gate.values())),readout_sha256=sha(__file__),features_sha256=sha(target/'features.npz'),
        qualification='Matched covariance baseline is a designed control; full means/distributions may remain informative. Conditional block intervals omit training and design uncertainty.'),indent=2)+'\n')
    print(score.round(4).to_string(index=False));print(gate)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['extract','analyze']);p.add_argument('--stage',choices=['development','final'],default='development');a=p.parse_args()
    with threadpool_limits(limits=4):globals()[a.action](a.stage)
