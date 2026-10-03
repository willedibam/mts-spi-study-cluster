"""Audit cached p90 and compare raw SPI means with SPI--SPI after raw calibration."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from scripts.calibrate_native_coupling import DATA,OUT
from scripts.spi_baseline_exploration import ROOT,sha,summarize,project_features
from src.corpus_geometry import fit_geometry_transform
from src.spi_spi_contract import build_unified_feature_values
from src.run_external_corpus import _array_sha256
from src.utils import slugify
from scripts.refresh_case_figures import style,save


def extract(data=DATA,out=OUT,corpus="native-gain-261003"):
    DATA,OUT=data,out
    manifest=json.loads((DATA/'manifest.json').read_text());rows=manifest['rows']
    means,distributions,zs,validity,profile_validity,sources=[],[],[],[],[],[];order=None
    with np.load(DATA/'observations.npz') as raw:
        assert sha(DATA/'observations.npz')==manifest['archive_sha256']
        for r in rows:
            folder=DATA/'mpis'/corpus/f"{r['corpus_index']+1:04d}-{slugify(r['row_id'],'dataset')}"
            meta=json.loads((folder/'meta.json').read_text());p=folder/'spi_mpis.npz'
            assert meta['status']=='complete' and meta['dataset_name']==r['row_id']
            assert meta['source']['archive_sha256']==manifest['archive_sha256']
            assert meta['source']['member_sha256']==_array_sha256(raw[r['row_id']])
            assert meta['pyspi']['config_sha256']==sha(ROOT/'configs/pyspi/benchmarked_p90.yaml')
            assert meta['job']['estimator_rng_policy']=='isolated_serial'
            names=[x['name'] for x in meta['pyspi']['spis']]
            if order is None:order=names
            assert order==names and len(names)==289
            with np.load(p) as a:mpis={k:a[k] for k in order}
            marginal=summarize(mpis,order)
            z,_,invalid=build_unified_feature_values(mpis,order)
            means.append(marginal[:,0]);distributions.append(marginal.reshape(-1));zs.append(z)
            validity.append(np.isfinite(marginal[:,0]).astype(float))
            profile_validity.append(np.array([k not in invalid for k in order],float))
            sources.append(dict(row_id=r['row_id'],mpi_sha256=sha(p),meta_sha256=sha(folder/'meta.json'),
                                compute_seconds=meta['job']['compute_seconds']))
    np.savez_compressed(OUT/'features.npz',mean=means,distribution=distributions,z=zs,validity=validity,profile_validity=profile_validity,z_validity=np.isfinite(zs).astype(np.uint8),
                        row_id=np.array([r['row_id'] for r in rows]),spi_order=np.array(order))
    (OUT/'sources.json').write_text(json.dumps(dict(sources=sources,manifest_sha256=sha(DATA/'manifest.json'),
        features_sha256=sha(OUT/'features.npz'),code_sha256=sha(Path(__file__))),indent=2)+'\n')


def scope_masks(rows):
    masks={f'all{rows.label.nunique()}':np.ones(len(rows),bool),
           f'CML{rows[rows.family.eq("CML")].label.nunique()}':rows.family.eq('CML').to_numpy()}
    if 'historical14' in rows:
        masks['historical14']=rows.historical14.to_numpy(bool)
        masks['coupled14']=rows.coupled.to_numpy(bool)
    return masks


def analyze(rows=None,out=OUT):
    OUT=out
    if rows is None:rows=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows'])
    scopes=scope_masks(rows)
    dev=rows.role.eq('development').to_numpy();held=~dev
    assert not set(rows.block[dev])&set(rows.block[held])
    with np.load(OUT/'features.npz') as a:
        np.testing.assert_array_equal(rows.row_id,a['row_id'])
        bank={k:a[k] for k in ['mean','distribution','z','validity','profile_validity','z_validity']}
    bank['strength_only']=rows.strength.to_numpy()[:,None]
    projections,diagnostics={},{}
    for name,x in bank.items():
        model,d,h=project_features(x[dev],x[held],dimensions=20)
        projections[name]=(d,h)
        missing=np.mean(~np.isfinite(x[:,model.transform.keep_indices]),axis=1)
        diagnostics[name]=dict(features=len(model.transform.keep_indices),components=d.shape[1],
            held_max_selected_missing_fraction=float(missing[held].max()),held_mean_selected_missing_fraction=float(missing[held].mean()),
            explained_variance=float(model.pca.explained_variance_ratio_.sum()))
    tr=fit_geometry_transform(bank['mean'][dev],scaling='standard',minimum_valid_fraction=.95)
    full=np.clip(tr.transform(bank['mean']),-5,5)
    projections['mean_without_PCA']=(full[dev],full[held])
    metrics,predictions=[],[]
    for name,(d,h) in projections.items():
        for scope,mask in scopes.items():
            a,b=mask[dev],mask[held]
            model=LogisticRegression(C=1,max_iter=3000).fit(d[a],rows.loc[dev,'label'].to_numpy()[a])
            pred=model.predict(h[b]);f=rows.loc[held].iloc[np.flatnonzero(b)][['row_id','label','block']].copy()
            f['predicted']=pred;f['correct']=f.label==pred;f['method']=name;f['scope']=scope
            predictions.append(f)
            group=f.groupby('block').correct.mean().to_numpy();rng=np.random.default_rng(261003)
            boot=group[rng.integers(len(group),size=(2000,len(group)))].mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
            metrics.append(dict(method=name,scope=scope,n=len(f),balanced_accuracy=balanced_accuracy_score(f.label,pred),
                                low=lo,high=hi,chance=1/f.label.nunique()))
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    predictions=pd.concat(predictions);predictions.to_csv(OUT/'predictions.csv',index=False)
    paired=[]
    for scope,mask in scopes.items():
        pivot=predictions[predictions.scope==scope].pivot(index=['row_id','block'],columns='method',values='correct').astype(float)
        for comparator in ['mean','mean_without_PCA','distribution']:
            delta=(pivot.z-pivot[comparator]).groupby('block').mean().to_numpy();rng=np.random.default_rng(261003)
            boot=delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(axis=1)
            lo,hi=np.quantile(boot,[.025,.975]);paired.append(dict(scope=scope,comparison='z - '+comparator,difference=delta.mean(),low=lo,high=hi))
    pd.DataFrame(paired).to_csv(OUT/'paired.csv',index=False)
    np.savez_compressed(OUT/'projections.npz',**{k+s:v for k,pair in projections.items() for s,v in zip(('_dev','_held'),pair)})
    (OUT/'analysis.json').write_text(json.dumps(dict(diagnostics=diagnostics,code_sha256=sha(Path(__file__)),
        features_sha256=sha(OUT/'features.npz'),status=f'exploratory {rows.label.nunique()}-condition generator-level gain comparison'),indent=2)+'\n')
    print(pd.DataFrame(metrics).round(4).to_string(index=False))


def plot_calibration(rows=None,out=OUT):
    OUT=out
    style();f=rows if rows is not None else pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows']);labels=sorted(f.label.unique())
    fig,axes=plt.subplots(1,2,figsize=(max(9,len(labels)*.8),4.6),layout='constrained')
    for i,label in enumerate(labels):
        g=f[f.label==label]
        color=plt.colormaps['tab20'].colors[i]
        axes[0].scatter(np.full(len(g),i),g.strength,s=13,alpha=.5,color=color)
        axes[1].scatter(np.full(len(g),i),g.mean_abs_Pearson,s=13,alpha=.5,color=color)
    axes[0].axhspan(.198,.202,color='.8',alpha=.25);axes[0].axhline(.2,color='.4',ls=':',lw=1)
    axes[0].set(ylabel='Mean total cross-channel Jacobian gain',title='Native-update gain'+(' (noise controls remain zero)' if f.family.eq('noise').any() else ''))
    axes[1].set(ylabel='Mean absolute Pearson correlation',title='Measured dependence need not match')
    for ax in axes:ax.set_xticks(range(len(labels)),labels,rotation=45,ha='right')
    return save(fig,OUT/'figures','calibration')


def plot_embedding(scope='all7',rows=None,out=OUT):
    OUT=out
    from umap import UMAP
    style();f=rows if rows is not None else pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows'])
    dev=f.role.eq('development').to_numpy();held=~dev
    mask=scope_masks(f)[scope];a,b=mask[dev],mask[held]
    labels=f.loc[held,'label'].to_numpy()[b];colors=dict(zip(sorted(f.label.unique()),plt.colormaps['tab20'].colors))
    fig,axes=plt.subplots(2,2,figsize=(8.3,8.5),layout='constrained')
    with np.load(OUT/'projections.npz') as p:
        for col,(name,title) in enumerate([('mean',r'Per-SPI means $m$'),('z',r'SPI–SPI $z$')]):
            d,h=p[name+'_dev'][a],p[name+'_held'][b]
            for row,kind in enumerate(['PCA','UMAP']):
                mapper=PCA(n_components=2) if kind=='PCA' else UMAP(n_neighbors=30,min_dist=.1,random_state=261003,transform_seed=261003,n_jobs=1)
                mapper.fit(d);xy=mapper.transform(h);ax=axes[row,col]
                for label in sorted(set(labels)):
                    keep=labels==label;ax.scatter(*xy[keep].T,s=20,color=colors[label],label=label,alpha=.65,linewidths=0)
                ax.set(title=title+f' — {len(set(labels))} classes',xlabel=kind+' 1',ylabel=kind+' 2');ax.set_box_aspect(1)
                for spine in ax.spines.values():spine.set_visible(True)
                if kind=='UMAP':ax.set(xticks=[],yticks=[])
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3 if len(set(labels))<10 else 4,fontsize=8)
    return save(fig,OUT/'figures','embedding-'+scope)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','analyze']);args=p.parse_args()
    with threadpool_limits(limits=4):{'extract':extract,'analyze':analyze}[args.stage]()
