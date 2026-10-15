"""Audited extraction and prespecified size-cell comparisons for strength-grid."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
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
from research.proof.baselines.strength_grid import DATA,OUT,RUN,CLASSES,ARMS,sha
from research.proof.baselines.analyze_strength_factorial import project,average_precision


def extract_one(task):
    data,row=task
    folder=Path(data)/'mpis'/slugify(row['cell_name'],'external-corpus')/f'{row["local_index"]:04d}-{slugify(row["row_id"],"dataset")}'
    meta=json.loads((folder/'meta.json').read_text())
    assert meta['status']=='complete' and meta['dataset_name']==row['row_id']
    assert (meta['M'],meta['T'])==(row['M'],row['T'])
    assert meta['source']['archive_sha256']==row['archive_sha256'] and meta['source']['member']==row['row_id']
    assert meta['random_seed']==2610152027 and not meta['normalise']
    names=[s['name'] for s in meta['pyspi']['spis']]
    with np.load(folder/'spi_mpis.npz') as data:
        vectors=np.array([((data[name]+data[name].T)/2)[np.triu_indices(row['M'],1)] for name in names])
    valid=np.isfinite(vectors).all(1);means=np.full(len(names),np.nan);means[valid]=vectors[valid].mean(1)
    z=_similarity_matrix(vectors,'pearson')[np.triu_indices(len(names),1)].astype(np.float32)
    identity=dict(catalogue=meta['pyspi']['config_sha256'],computation=meta['pyspi']['version']['computation'],source=meta['pyspi']['version']['source'],names=names)
    quality=dict(row_id=row['row_id'],cell=row['cell'],label=row['label'],arm=row['arm'],finite_means=int(np.isfinite(means).sum()),finite_z=int(np.isfinite(z).sum()),errors=json.dumps(meta['pyspi'].get('errors',{})),compute_seconds=meta['job'].get('compute_seconds'))
    return means,z,identity,quality


def extract(data,out,workers):
    manifest=json.loads((data/'manifest.json').read_text());rows=manifest['rows'];assert len(rows)==1350
    for cell in manifest['cells']:assert sha(data/(cell['cell']+'.npz'))==cell['archive_sha256']
    means=[];zs=[];qualities=[];identity=None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(m,z,current,quality) in enumerate(pool.map(extract_one,[(str(data),r) for r in rows],chunksize=4)):
            identity=identity or current;assert current==identity
            means.append(m);zs.append(z);qualities.append(quality)
            if (i+1)%150==0:print('Extracted',i+1,flush=True)
    assert len(identity['names'])==289 and identity['computation']=='3.0.0.r9'
    assert identity['catalogue']=='bc4bafa16b4add8bb6283db490a9fa3d8dde4fde47093147d32579b989397965'
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'features.npz',mean=np.array(means),z=np.array(zs),row_id=np.array([r['row_id'] for r in rows]),spi_order=np.array(identity.pop('names')))
    pd.DataFrame(qualities).to_csv(out/'estimator-quality.csv',index=False)
    (out/'extraction-audit.json').write_text(json.dumps(dict(records=1350,features_sha256=sha(out/'features.npz'),manifest_sha256=sha(data/'manifest.json'),source_sha256=sha(__file__),**identity),indent=2)+'\n')


def split(rows):
    core=rows.role.eq('core').to_numpy()
    train=core & rows.instance.lt(3).to_numpy() & rows.arm.isin(['fixed_clean','fixed_sensor']).to_numpy()
    evaluation=core & rows.instance.ge(3).to_numpy()
    gallery=train & rows.arm.eq('fixed_clean').to_numpy()
    assert (train.sum(),evaluation.sum(),gallery.sum())==(378,504,189)
    assert set(rows.loc[train,'parent']).isdisjoint(rows.loc[evaluation,'parent'])
    return train,evaluation,gallery


def ratio(values,labels):
    centers=np.array([values[labels==c].mean(0) for c in CLASSES])
    within=np.mean([np.mean(np.sum((values[labels==c]-centers[i])**2,1)) for i,c in enumerate(CLASSES)])
    between=np.mean(np.sum((centers-centers.mean(0))**2,1))
    return float(within/between)


def analyze(data,out):
    rows=pd.DataFrame(json.loads((data/'manifest.json').read_text())['rows'])
    audit=json.loads((out/'extraction-audit.json').read_text());assert sha(out/'features.npz')==audit['features_sha256']
    with np.load(out/'features.npz') as a:bank={k:a[k] for k in a.files}
    np.testing.assert_array_equal(bank['row_id'],rows.row_id)
    train,evaluation,gallery=split(rows);labels=rows.label.to_numpy();spaces={};details=[];missing=[]
    for name,key,standard,valid in [('m','mean',True,.95),('z','z',False,.95),('z_standard','z',True,.95),('z_complete','z',False,1.),('m_unclipped','mean',True,.95)]:
        full,pcs,transform,pca=project(bank[key],train,standard,valid,clip=name!='m_unclipped')
        spaces[name]=pcs
        if name in ['m','m_unclipped']:spaces[name+'_full']=full
        details.append(dict(representation=name,features=len(transform.keep_indices),PCs=pcs.shape[1],training_variance_retained=float(pca.explained_variance_ratio_.sum())))
        for (cell,arm,label),indices in rows[evaluation].groupby(['cell','arm','label']).groups.items():
            missing.append(dict(representation=name,cell=cell,arm=arm,label=label,fraction=float((~np.isfinite(bank[key][indices][:,transform.keep_indices])).mean())))
    fused=np.column_stack([spaces[k]/np.sqrt(spaces[k][train].var(0).sum()) for k in ['m','z']])
    spaces['m+z']=PCA(50,svd_solver='full').fit(fused[train]).transform(fused)
    scalar=rows.mean_abs_r.to_numpy()[:,None];spaces['mean_abs_r']=(scalar-scalar[train].mean(0))/scalar[train].std(0)
    np.savez_compressed(out/'coordinates.npz',row_id=rows.row_id.to_numpy(dtype=str),**spaces)
    scores=[];individual=[]
    for name,values in spaces.items():
        model=LogisticRegression(C=1,max_iter=5000).fit(values[train],labels[train])
        rbf=SVC(C=1).fit(values[train],labels[train]) if name.endswith('_full') else None
        for arm in ARMS:
            choose=evaluation & rows.arm.eq(arm).to_numpy();y=labels[choose];v=values[choose]
            ap,top=average_precision(cdist(v,values[gallery]),y[:,None]==labels[gallery][None,:])
            predicted=model.predict(v)
            f=rows.loc[choose,['row_id','parent','cell','M','T','instance','label','arm']].copy();f['representation']=name;f['AP']=ap;f['top1']=top;f['correct']=predicted==y;f['prediction']=predicted;individual.append(f)
            masks=[('all',np.ones(len(f),bool))]+[(cell,f.cell.eq(cell).to_numpy()) for cell in rows.cell.unique()]
            for cell,mask in masks:
                entry=dict(representation=name,arm=arm,cell=cell,n=int(mask.sum()),spread_to_separation=ratio(v[mask],y[mask]),silhouette=silhouette_score(v[mask],y[mask]),mAP=float(ap[mask].mean()),top1=float(top[mask].mean()),accuracy=float((predicted[mask]==y[mask]).mean()))
                if rbf is not None:entry['RBF_accuracy']=float(np.mean(rbf.predict(v[mask])==y[mask]))
                scores.append(entry)
    pd.DataFrame(scores).to_csv(out/'metrics.csv',index=False);pd.concat(individual).to_csv(out/'individual.csv',index=False)
    pd.DataFrame(details).to_csv(out/'preprocessing.csv',index=False);pd.DataFrame(missing).to_csv(out/'missingness.csv',index=False)
    validity=np.isfinite(bank['z']).astype(float);keep=validity[train].std(0)>0
    model=LogisticRegression(C=1,max_iter=5000).fit(validity[train][:,keep],labels[train]) if keep.any() else None
    validity_scores={arm:float(np.mean(model.predict(validity[use][:,keep])==labels[use])) if model else 1/7 for arm in ARMS for use in [evaluation & rows.arm.eq(arm).to_numpy()]}
    report=dict(training=378,evaluation=504,gallery=189,classes=CLASSES,validity_only_accuracy=validity_scores,source_sha256=sha(__file__),notes=['Independent parent in every size cell; paired interventions within parent.','Only 2 evaluation instances/class/cell: individual cell geometry is descriptive.','PCA/UMAP training fixed conditions only; t-SNE is separate joint descriptive fit.','No distribution summaries or within-recording normalization of mean vectors.'])
    (out/'analysis-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(scores).query("cell == 'all' and representation in ['m','z','m+z']").round(3).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['extract','analyze']);parser.add_argument('--data',type=Path,default=DATA);parser.add_argument('--out',type=Path,default=OUT);parser.add_argument('--workers',type=int,default=4);a=parser.parse_args()
    with threadpool_limits(limits=1 if a.stage=='extract' else 4):extract(a.data,a.out,a.workers) if a.stage=='extract' else analyze(a.data,a.out)
