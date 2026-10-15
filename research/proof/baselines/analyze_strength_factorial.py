"""Frozen-split geometry and transfer checks for the paired strength factorial."""
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

from mtsspi.corpus_geometry import fit_geometry_transform
from mtsspi.spi_spi_contract import _similarity_matrix
from mtsspi.utils import slugify
from research.proof.baselines.strength_factorial import ROOT, DATA, OUT, RUN, CLASSES, ARMS, sha


def extract_one(task):
    data,row,archive_hash=task
    folder=Path(data)/'mpis'/RUN/f'{row["corpus_index"]+1:04d}-{slugify(row["row_id"],"dataset")}'
    meta=json.loads((folder/'meta.json').read_text())
    assert meta['status']=='complete' and meta['dataset_name']==row['row_id']
    assert (meta['M'],meta['T'])==(16,1000)
    assert meta['source']['archive_sha256']==archive_hash
    assert meta['random_seed']==2610152027
    names=[s['name'] for s in meta['pyspi']['spis']]
    with np.load(folder/'spi_mpis.npz') as a:
        vectors=np.array([((a[name]+a[name].T)/2)[np.triu_indices(16,1)] for name in names])
    valid=np.isfinite(vectors).all(1)
    means=np.full(len(names),np.nan);means[valid]=vectors[valid].mean(1)
    z=_similarity_matrix(vectors,'pearson')[np.triu_indices(len(names),1)].astype(np.float32)
    lineage=dict(catalogue=meta['pyspi']['config_sha256'],computation=meta['pyspi']['version']['computation'],
                 n_spis=len(names),spi_order=names)
    return means,z,lineage


def extract(data,out,workers):
    manifest=json.loads((data/'manifest.json').read_text());rows=manifest['rows']
    assert len(rows)==1200 and sha(data/'observations.npz')==manifest['archive_sha256']
    lineage=None;means=[];zs=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(m,z,current) in enumerate(pool.map(extract_one,[(str(data),r,manifest['archive_sha256']) for r in rows],chunksize=4)):
            lineage=lineage or current
            assert current==lineage,'Mixed catalogue/computation version/order'
            means.append(m);zs.append(z)
            if (i+1)%200==0:print('Extracted',i+1,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'features.npz',mean=np.array(means),z=np.array(zs),row_id=np.array([r['row_id'] for r in rows]),spi_order=np.array(lineage.pop('spi_order')))
    audit=dict(records=len(rows),source_archive_sha256=manifest['archive_sha256'],features_sha256=sha(out/'features.npz'),
               extraction_source_sha256=sha(__file__),**lineage)
    (out/'extraction-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(audit)


def split(rows):
    core=rows.label.isin(CLASSES).to_numpy()
    train=core & rows.instance.lt(20).to_numpy() & rows.arm.isin(['fixed_clean','fixed_sensor']).to_numpy()
    evaluation=core & rows.instance.ge(20).to_numpy()
    gallery=core & rows.instance.lt(20).to_numpy() & rows.arm.eq('fixed_clean').to_numpy()
    assert train.sum()==280 and evaluation.sum()==560 and gallery.sum()==140
    assert set(rows.loc[train,'instance']).isdisjoint(rows.loc[evaluation,'instance'])
    assert not rows.loc[train,'arm'].str.startswith('varying').any()
    return train,evaluation,gallery


def project(raw,train,standard,valid=.95):
    transform=fit_geometry_transform(raw[train],scaling='standard' if standard else 'center',minimum_valid_fraction=valid)
    values=transform.transform(raw)
    if standard:values=np.clip(values,-5,5)
    pca=PCA(n_components=min(50,train.sum()-1,values.shape[1]),svd_solver='randomized',random_state=261015).fit(values[train])
    return values,pca.transform(values),transform,pca


def average_precision(distances,relevant):
    ranking=np.argsort(distances,axis=1,kind='stable')
    hits=np.take_along_axis(relevant,ranking,axis=1)
    precision=hits.cumsum(1)/np.arange(1,hits.shape[1]+1)
    return (precision*hits).sum(1)/hits.sum(1),hits[:,0]


def variance_ratio(values,labels):
    centers=np.array([values[labels==c].mean(0) for c in CLASSES])
    within=np.array([np.mean(np.sum((values[labels==c]-centers[i])**2,axis=1)) for i,c in enumerate(CLASSES)])
    between=float(np.mean(np.sum((centers-centers.mean(0))**2,axis=1)))
    return float(within.mean()/between),within,between


def analyze(data,out):
    rows=pd.DataFrame(json.loads((data/'manifest.json').read_text())['rows'])
    audit=json.loads((out/'extraction-audit.json').read_text());assert sha(out/'features.npz')==audit['features_sha256']
    with np.load(out/'features.npz') as archive:
        bank={k:archive[k] for k in archive.files}
    np.testing.assert_array_equal(bank['row_id'],rows.row_id)
    train,evaluation,gallery=split(rows);labels=rows.label.to_numpy()
    spaces={};details=[];missing=[]
    for name,key,standard,valid in [('m','mean',True,.95),('z','z',False,.95),('z_standard','z',True,.95),('z_complete','z',False,1.)]:
        full,pcs,transform,pca=project(bank[key],train,standard,valid)
        spaces[name]=pcs
        if name=='m':spaces['m_full']=full
        details.append(dict(representation=name,features=len(transform.keep_indices),PCs=pcs.shape[1],training_variance_retained=pca.explained_variance_ratio_.sum()))
        for arm in ARMS:
            for label in CLASSES:
                select=evaluation & rows.arm.eq(arm).to_numpy() & rows.label.eq(label).to_numpy()
                missing.append(dict(representation=name,arm=arm,label=label,fraction=float((~np.isfinite(bank[key][select][:,transform.keep_indices])).mean())))
    blocks=[spaces[k]/np.sqrt(spaces[k][train].var(0).sum()) for k in ['m','z']]
    concatenated=np.column_stack(blocks)
    fusion=PCA(50,svd_solver='full').fit(concatenated[train]);spaces['m+z']=fusion.transform(concatenated)
    scalar=rows.mean_abs_r.to_numpy()[:,None]
    spaces['mean_abs_r']=(scalar-scalar[train].mean(0))/scalar[train].std(0)
    coordinates=dict(row_id=bank['row_id'],**spaces)
    np.savez_compressed(out/'coordinates.npz',**coordinates)
    scores=[];individual=[];per_class=[]
    for name,values in spaces.items():
        model=LogisticRegression(C=1,max_iter=5000).fit(values[train],labels[train])
        # Includes a strong raw-mean readout without PCA, alongside the common linear readout.
        rbf=SVC(C=1).fit(values[train],labels[train]) if name=='m_full' else None
        for arm in ARMS:
            choose=evaluation & rows.arm.eq(arm).to_numpy();y=labels[choose];v=values[choose]
            distances=cdist(v,values[gallery]);ap,top=average_precision(distances,y[:,None]==labels[gallery][None,:])
            prediction=model.predict(v);ratio,within,between=variance_ratio(v,y)
            score=dict(representation=name,arm=arm,spread_to_separation=ratio,within_variance=within.mean(),between_variance=between,
                       silhouette=silhouette_score(v,y),mAP=ap.mean(),top1=top.mean(),accuracy=np.mean(prediction==y))
            if rbf is not None:score['RBF_accuracy']=np.mean(rbf.predict(v)==y)
            scores.append(score)
            f=rows.loc[choose,['row_id','instance','label','arm']].copy();f['representation']=name;f['AP']=ap;f['top1']=top;f['prediction']=prediction;individual.append(f)
            for i,label in enumerate(CLASSES):
                per_class.append(dict(representation=name,arm=arm,label=label,within_variance=within[i],spread_to_separation=within[i]/between,mAP=ap[y==label].mean(),accuracy=np.mean(prediction[y==label]==label)))
    valid=np.isfinite(bank['z']).astype(float);keep=valid[train].std(0)>0
    if keep.any():
        model=LogisticRegression(C=1,max_iter=5000).fit(valid[train][:,keep],labels[train])
        validity={arm:float(np.mean(model.predict(valid[evaluation & rows.arm.eq(arm).to_numpy()][:,keep])==labels[evaluation & rows.arm.eq(arm).to_numpy()])) for arm in ARMS}
    else:validity={arm:1/len(CLASSES) for arm in ARMS}
    score_frame=pd.DataFrame(scores);individual=pd.concat(individual)
    score_frame.to_csv(out/'metrics.csv',index=False);individual.to_csv(out/'individual.csv',index=False)
    pd.DataFrame(per_class).to_csv(out/'per-class.csv',index=False);pd.DataFrame(details).to_csv(out/'preprocessing.csv',index=False);pd.DataFrame(missing).to_csv(out/'missingness.csv',index=False)
    rng=np.random.default_rng(261015);draws=rng.integers(0,20,size=(2000,20));intervals=[]
    for arm in ARMS:
        paired=individual[(individual.arm==arm)&individual.representation.isin(['m','z','m+z'])].groupby(['instance','representation']).AP.mean().unstack()
        for left,right in [('z','m'),('m+z','m')]:
            difference=(paired[left]-paired[right]).to_numpy();boot=difference[draws].mean(1)
            intervals.append(dict(arm=arm,contrast=left+' minus '+right,metric='mAP',difference=difference.mean(),low=np.quantile(boot,.025),high=np.quantile(boot,.975)))
        # Recompute geometry in each paired bootstrap, holding the fitted representation fixed.
        tensors={}
        for name in ['m','z']:
            tensors[name]=np.array([spaces[name][evaluation & rows.arm.eq(arm).to_numpy() & rows.label.eq(c).to_numpy()] for c in CLASSES])
        bootstrap=[]
        for indices in draws:
            ratios=[]
            for name in ['m','z']:
                a=tensors[name][:,indices,:];centers=a.mean(1)
                w=((a-centers[:,None,:])**2).sum(2).mean()
                b=((centers-centers.mean(0))**2).sum(1).mean()
                ratios.append(w/b)
            bootstrap.append(ratios[1]-ratios[0])
        observed=score_frame[score_frame.arm.eq(arm)].set_index('representation').spread_to_separation
        intervals.append(dict(arm=arm,contrast='z minus m',metric='spread_to_separation',difference=observed['z']-observed['m'],low=np.quantile(bootstrap,.025),high=np.quantile(bootstrap,.975)))
    pd.DataFrame(intervals).to_csv(out/'paired-intervals.csv',index=False)
    report=dict(training=280,evaluation=560,gallery=140,classes=list(CLASSES),validity_only_accuracy=validity,
                catalogue_size=int(audit['n_spis']),computation=audit['computation'],source_sha256=sha(__file__),
                notes=['All preprocessing/PCA fitted only on fixed clean/sensor training instances0-19.',
                       'Evaluation instances20-39 are disjoint; every retrieval gallery item has a different parent.',
                       'm is raw MPI means with training feature scaling/clipping; no within-recording mean-vector normalization.',
                       'Bootstrap intervals are conditional on fitted training data and retain class/arm pairing.',
                       'z is symmetrized to match the historical proof; no directional-mechanism claim.'])
    (out/'analysis-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(score_frame[score_frame.representation.isin(['m','z','m+z'])].round(4).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','analyze']);p.add_argument('--data',type=Path,default=DATA);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    with threadpool_limits(limits=1 if a.stage=='extract' else 4):
        extract(a.data,a.out,a.workers) if a.stage=='extract' else analyze(a.data,a.out)
