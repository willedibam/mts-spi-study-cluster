"""Audited fixed-readout analysis for paired noise strength and spatial pattern."""
import argparse,json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from mtsspi.spi_spi_contract import _similarity_matrix
from mtsspi.utils import slugify
from research.proof.baselines.noise_experiments import DATA,OUT,ROOT,CLASSES,ETAS,PYSPI,sha
from research.proof.baselines.analyze_strength_factorial import project,average_precision

CONDITIONS=['clean']+[f'{p}-eta{e:g}' for e in ETAS[1:] for p in ['equal','exponential']]+['noise-limit','random-previous','random-stronger']

def extract_one(task):
    row,config_hashes,code_hashes=task
    folder=Path(row['mpi_root'])/row['corpus_name']/f'{row["local_index"]:04d}-{slugify(row["row_id"],"dataset")}'
    meta=json.loads((folder/'meta.json').read_text())
    assert meta['status']=='complete' and meta['dataset_name']==row['row_id'] and (meta['M'],meta['T'])==(16,1000)
    assert meta['source']['archive_sha256']==row['archive_sha256'] and meta['source']['member']==row['row_id']
    assert meta['random_seed']==2610152027 and not meta['normalise']
    assert meta['execution_identity']['corpus_config_sha256']==config_hashes[row['experiment']]
    assert meta['execution_identity']['code_sha256']==code_hashes
    assert meta['job']['estimator_rng_policy']=='pyspi_fixed_seeds'
    version=meta['pyspi']['version'];assert version['source']==PYSPI+'/pyspi' and version['computation']=='3.0.0.r9'
    assert meta['pyspi']['config_sha256']=='bc4bafa16b4add8bb6283db490a9fa3d8dde4fde47093147d32579b989397965'
    names=[s['name'] for s in meta['pyspi']['spis']];assert len(names)==289
    errors=meta['pyspi'].get('errors',{})
    assert not any(any(t in e for t in ['ImportError:','ModuleNotFoundError:','AttributeError:','NameError:','TypeError:','KeyError:']) for e in errors.values())
    with np.load(folder/'spi_mpis.npz') as a:
        raw=np.array([a[n] for n in names]);vectors=((raw+raw.transpose(0,2,1))/2)[:,np.triu_indices(16,1)[0],np.triu_indices(16,1)[1]]
    means=vectors.mean(1);means[~np.isfinite(vectors).all(1)]=np.nan
    z=_similarity_matrix(vectors,'pearson')[np.triu_indices(289,1)].astype(np.float32)
    # Independent numeric audit on all pilots, chosen before SPI outcomes.
    numeric=None
    if row['instance']==0:
        np.testing.assert_allclose(raw[:,~np.eye(16,dtype=bool)].mean(1),means,rtol=1e-10,atol=1e-10,equal_nan=True)
        pairs=np.triu_indices(289,1);maxerr=0.;undefined=0
        for k in np.linspace(0,len(z)-1,100,dtype=int):
            u,v=vectors[pairs[0][k]],vectors[pairs[1][k]]
            if not (np.isfinite(u).all() and np.isfinite(v).all()) or np.linalg.norm(u-u.mean())<1e-12 or np.linalg.norm(v-v.mean())<1e-12:
                assert not np.isfinite(z[k]);undefined+=1
            else:
                err=abs(np.corrcoef(u,v)[0,1]-z[k]);assert err<1e-6;maxerr=max(maxerr,err)
        numeric=dict(row_id=row['row_id'],mean_checks=289,z_checks=100,max_error=maxerr,undefined=undefined)
    quality=dict(row_id=row['row_id'],label=row['label'],arm=row['arm'],finite_means=int(np.isfinite(means).sum()),finite_z=int(np.isfinite(z).sum()),errors=json.dumps(errors),compute_seconds=meta['job']['compute_seconds'])
    return means,z,names,quality,numeric

def extract(data,out,workers):
    manifest=json.loads((data/'manifest.json').read_text());rows=manifest['rows'];assert len(rows)==3136
    config_hashes={k:sha(data/(k+'.yaml')) for k in ['dose','random']}
    for k in config_hashes:assert sha(data/(k+'.npz'))==next(r['archive_sha256'] for r in rows if r['experiment']==k)
    code_hashes={n:sha(ROOT/'src/mtsspi'/n) for n in ['run_corpus.py','corpus.py','generated_corpus.py','compute.py']}
    means=[];zs=[];qualities=[];numeric=[];names=None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(m,z,current,q,a) in enumerate(pool.map(extract_one,[(r,config_hashes,code_hashes) for r in rows],chunksize=4)):
            if names is None:names=current
            assert current==names;means.append(m);zs.append(z);qualities.append(q)
            if a is not None:numeric.append(a)
            if (i+1)%224==0:print('Extracted',i+1,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'features.npz',mean=np.array(means),z=np.array(zs),row_id=np.array([r['row_id'] for r in rows]),spi_order=np.array(names))
    pd.DataFrame(qualities).to_csv(out/'estimator-quality.csv',index=False);pd.DataFrame(numeric).to_csv(out/'independent-values.csv',index=False)
    audit=dict(records=len(rows),features_sha256=sha(out/'features.npz'),manifest_sha256=sha(data/'manifest.json'),source_sha256=sha(__file__),code_hashes=code_hashes,catalogue_size=289,pyspi_source=PYSPI,computation='3.0.0.r9',independent_records=len(numeric),max_independent_error=max(r['max_error'] for r in numeric))
    (out/'extraction-audit.json').write_text(json.dumps(audit,indent=2)+'\n')

def split(rows):
    train=rows.instance.lt(16).to_numpy() & rows.arm.isin(['clean','equal-eta1','exponential-eta1']).to_numpy()
    evaluation=rows.instance.ge(16).to_numpy();gallery=train & rows.arm.eq('clean').to_numpy()
    assert (train.sum(),evaluation.sum(),gallery.sum())==(336,1568,112)
    assert set(rows.loc[train,'parent']).isdisjoint(rows.loc[evaluation,'parent'])
    return train,evaluation,gallery

def geometry(v,y):
    centers=np.array([v[y==c].mean(0) for c in CLASSES])
    within=float(np.mean([np.square(v[y==c]-centers[i]).sum(1).mean() for i,c in enumerate(CLASSES)]))
    between=float(np.square(centers-centers.mean(0)).sum(1).mean())
    return within/between,within,between

def analyze(data,out):
    rows=pd.DataFrame(json.loads((data/'manifest.json').read_text())['rows']);audit=json.loads((out/'extraction-audit.json').read_text())
    assert sha(out/'features.npz')==audit['features_sha256'] and sha(data/'manifest.json')==audit['manifest_sha256']
    with np.load(out/'features.npz') as a:bank={k:a[k] for k in a.files}
    np.testing.assert_array_equal(bank['row_id'],rows.row_id)
    train,evaluation,gallery=split(rows);y=rows.label.to_numpy();spaces={};details=[];missing=[]
    for name,key,standard,valid,clip in [('m','mean',True,.95,True),('z','z',False,.95,False),('z_standard','z',True,.95,True),('z_complete','z',False,1.,False),('m_unclipped','mean',True,.95,False)]:
        full,pcs,tr,pca=project(bank[key],train,standard,valid,clip=clip);spaces[name]=pcs
        if name in ['m','m_unclipped']:spaces[name+'_full']=full
        details.append(dict(representation=name,features=len(tr.keep_indices),PCs=pcs.shape[1],retained_training_variance=pca.explained_variance_ratio_.sum()))
        for (arm,label),ix in rows[evaluation].groupby(['arm','label']).groups.items():
            missing.append(dict(representation=name,arm=arm,label=label,fraction=float((~np.isfinite(bank[key][ix][:,tr.keep_indices])).mean())))
    anchor=spaces['m_full'][rows.instance.lt(16) & rows.arm.eq('noise-limit')].mean(0)
    direction=spaces['m_full']-anchor;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
    spaces['m_direction']=PCA(50,svd_solver='full').fit(direction[train]).transform(direction)
    scalar=rows.mean_abs_r.to_numpy()[:,None];spaces['mean_abs_r']=(scalar-scalar[train].mean())/scalar[train].std()
    np.savez_compressed(out/'coordinates.npz',row_id=bank['row_id'],**spaces)
    scores=[];individual=[];per_class=[]
    clean_index=rows[rows.arm.eq('clean')].set_index(['label','instance']).index
    lookup={tuple(k):idx for k,idx in zip(clean_index,rows.index[rows.arm.eq('clean')])}
    for name,v in spaces.items():
        model=LogisticRegression(C=1,max_iter=5000).fit(v[train],y[train]);rbf=SVC(C=1).fit(v[train],y[train]) if name.endswith('_full') else None
        scale=np.sqrt(geometry(v[gallery],y[gallery])[2])
        for arm in CONDITIONS:
            use=evaluation & rows.arm.eq(arm).to_numpy();a=v[use];labels=y[use]
            distances=cdist(a,v[gallery])
            ap,top=average_precision(distances,labels[:,None]==y[gallery][None,:]);pred=model.predict(a)
            # Exact expectation for a uniformly randomized query family, holding
            # each distance ranking fixed. AP's null is not classification chance.
            null_ap=np.mean([average_precision(distances,np.broadcast_to(y[gallery]==c,distances.shape))[0] for c in CLASSES],axis=0)
            ratio,w,b=geometry(a,labels)
            matched=np.array([lookup[(r.label,r.instance)] for r in rows[use].itertuples()]);movement=np.linalg.norm(a-v[matched],axis=1)/scale
            entry=dict(representation=name,arm=arm,n=len(a),spread_to_separation=ratio,within_variance=w,between_variance=b,silhouette=silhouette_score(a,labels),mAP=float(ap.mean()),top1=float(top.mean()),accuracy=float(np.mean(pred==labels)),paired_movement=float(movement.mean()))
            entry.update(mAP_label_null=float(null_ap.mean()),mAP_excess=float((ap-null_ap).mean()))
            if rbf is not None:entry['RBF_accuracy']=float(np.mean(rbf.predict(a)==labels))
            scores.append(entry)
            f=rows.loc[use,['row_id','instance','parent','label','arm']].copy();f['representation']=name;f['AP']=ap;f['AP_label_null']=null_ap;f['correct']=pred==labels;f['movement']=movement;individual.append(f)
            for c in CLASSES:per_class.append(dict(representation=name,arm=arm,label=c,mAP=float(ap[labels==c].mean()),accuracy=float(np.mean(pred[labels==c]==c))))
    scores=pd.DataFrame(scores);individual=pd.concat(individual);scores.to_csv(out/'metrics.csv',index=False);individual.to_csv(out/'individual.csv',index=False)
    pd.DataFrame(per_class).to_csv(out/'per-class.csv',index=False);pd.DataFrame(details).to_csv(out/'preprocessing.csv',index=False);pd.DataFrame(missing).to_csv(out/'missingness.csv',index=False)
    valid=np.isfinite(bank['z']).astype(np.float32);keep=valid[train].std(0)>0
    model=LogisticRegression(C=1,max_iter=5000).fit(valid[train][:,keep],y[train]) if keep.any() else None
    validity={arm:float(np.mean(model.predict(valid[use][:,keep])==y[use])) if model else 1/7 for arm in CONDITIONS for use in [evaluation & rows.arm.eq(arm).to_numpy()]}
    draws=np.random.default_rng(261016).integers(0,16,size=(2000,16));intervals=[];boot_geometry={}
    for arm in CONDITIONS:
        frame=individual[individual.arm.eq(arm)&individual.representation.isin(['m','z'])].groupby(['instance','representation']).AP.mean().unstack();delta=(frame.z-frame.m).to_numpy();b=delta[draws].mean(1)
        intervals.append(dict(arm=arm,metric='z_minus_m_mAP',estimate=float(delta.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975))))
        tensors={name:np.array([spaces[name][evaluation & rows.arm.eq(arm).to_numpy() & rows.label.eq(c).to_numpy()] for c in CLASSES]) for name in ['m','z']}
        ratios=[]
        for indices in draws:
            pair=[]
            for name in ['m','z']:
                a=tensors[name][:,indices];centers=a.mean(1);pair.append(float(np.square(a-centers[:,None]).sum(2).mean()/np.square(centers-centers.mean(0)).sum(1).mean()))
            ratios.append(pair)
        boot_geometry[arm]=np.array(ratios)
    point=scores[scores.representation.isin(['m','z'])].pivot(index='arm',columns='representation',values='spread_to_separation')
    for arm in CONDITIONS:
        b=boot_geometry[arm]-boot_geometry['clean'];d=b[:,1]-b[:,0];p=point.loc[arm]-point.loc['clean']
        intervals.append(dict(arm=arm,metric='z_minus_m_change_in_spread_ratio',estimate=float(p.z-p.m),low=float(np.quantile(d,.025)),high=float(np.quantile(d,.975))))
    pd.DataFrame(intervals).to_csv(out/'paired-intervals.csv',index=False)
    (out/'analysis-audit.json').write_text(json.dumps(dict(training=336,evaluation=1568,gallery=112,validity_only_accuracy=validity,source_sha256=sha(__file__),notes=['No outcome-selected doses/classes/seeds.','Primary m has no per-recording normalization; m_direction is secondary, anchored to training pure-noise mean then L2 normalized.','Matched expected total noise power is not matched average pair attenuation.','Intervals resample instance blocks keeping classes/conditions paired; conditional on training.']),indent=2)+'\n')
    print(scores.query("representation in ['m','z']").round(3).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','analyze']);p.add_argument('--data',type=Path,default=DATA);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    with threadpool_limits(limits=1 if a.stage=='extract' else 4):extract(a.data,a.out,a.workers) if a.stage=='extract' else analyze(a.data,a.out)
