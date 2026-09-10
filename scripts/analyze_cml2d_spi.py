"""Frozen PC1 exploratory order tracking, using all p90 MPIs; no SPI extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.spi_spi_contract import build_unified_feature_values, UNIFIED_CONTRACT_VERSION
from src.spi_spi_analysis import fit_feature_transform
from src.utils import slugify


def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corr(x,y):
    return float(pd.Series(np.asarray(x)).corr(pd.Series(np.asarray(y)),method='spearman'))


def assemble(corpus, mpi_root):
    manifest=json.loads((corpus/'manifest.json').read_text()); rows=manifest['rows']; values=[];sources=[];order=None
    for row in rows:
        folder=mpi_root/f"{row['corpus_index']:04d}-{slugify(row['row_id'],'dataset')}"
        meta=json.loads((folder/'meta.json').read_text())
        assert meta['status']=='complete' and meta['dataset_name']==row['row_id']
        assert meta['M']==row['M'] and meta['T']==row['T']
        assert meta['source']['archive_sha256']==manifest['archive_sha256']
        current=[s['name'] for s in meta['pyspi']['spis']]
        assert len(current)==289
        if order is None: order=current
        assert order==current
        with np.load(folder/'spi_mpis.npz',allow_pickle=False) as a:
            z,_,_=build_unified_feature_values(a,order)
        values.append(z);sources.append(dict(row_id=row['row_id'],mpi_sha256=file_hash(folder/'spi_mpis.npz'),
            config_sha256=meta['pyspi']['config_sha256'],execution_identity=meta['execution_identity']))
    assert len({s['config_sha256'] for s in sources})==1
    return rows,np.asarray(values),order,sources


def run(corpus,mpi_root,output,frozen=None):
    if output.exists():raise FileExistsError(output)
    rows,z,order,sources=assemble(corpus,mpi_root)
    frame=pd.DataFrame([{k:r[k] for k in ['row_id','r','L','N','seed','M','T','view','role']} for r in rows])
    dev=(frame.role=='development').to_numpy(); held=~dev
    if frozen:
        with np.load(frozen/'model.npz',allow_pickle=False) as a:
            assert a['spi_order'].tolist()==order
            keep=a['keep'];impute=a['impute'];center=a['center'];component=a['component'];score_scale=float(a['score_scale'])
        geometry=json.loads((frozen/'geometry.json').read_text())
    else:
        transform=fit_feature_transform(z[dev],['unified']*z.shape[1],minimum_valid_fraction=.99,variance_threshold=.05,block_balanced=False)
        keep=transform.keep_indices;impute=transform.impute_values;center=transform.center
        training=transform.transform(z[dev]);pca=PCA(n_components=min(5,len(training)-1),svd_solver='full').fit(training)
        component=pca.components_[0].copy()
        if component[np.argmax(abs(component))]<0:component*=-1
        score_scale=float(np.std(training@component))
        # Target-blind stability by leaving out each independent development seed.
        cosines=[]
        for seed in sorted(frame.loc[dev,'seed'].unique()):
            subset=training[frame.loc[dev,'seed'].to_numpy()!=seed]
            v=PCA(n_components=1,svd_solver='full').fit(subset).components_[0]
            cosines.append(float(abs(v@component)))
        evr=pca.explained_variance_ratio_.tolist()
        geometry=dict(evr=evr,leave_seed_loading_cosines=cosines,
            passes_one_coordinate_gate=bool(evr[0]>=.2 and evr[0]/evr[1]>=1.5 and min(cosines)>=.8))
    missing=np.mean(~np.isfinite(z[:,keep]),axis=1)
    eligible=missing<=.05
    q=((np.where(np.isfinite(z[:,keep]),z[:,keep],impute)-center)@component)/score_scale
    q[~eligible]=np.nan
    frame['q']=q;frame['selected_missingness']=missing;frame['eligible']=eligible
    output.mkdir(parents=True)
    np.savez_compressed(output/'model.npz',keep=keep,impute=impute,center=center,component=component,score_scale=score_scale,spi_order=np.array(order))
    np.savez_compressed(output/'features.npz',z=z,row_id=np.array(frame.row_id),spi_order=np.array(order))
    (output/'geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
    # Seal fit/eligibility before reading physical targets into the score table.
    counts=frame.groupby(['role','view','M','T','r']).eligible.agg(['sum','size'])
    valid_gate=bool(np.mean(~eligible)<=.1 and (counts['sum']>=2).all())
    identity=dict(contract=UNIFIED_CONTRACT_VERSION,manifest_sha256=file_hash(corpus/'manifest.json'),
        sources=sources,code_sha256=file_hash(__file__),selected_features=len(keep),
        passes_row_gate=valid_gate,geometry=geometry,
        fit_uses_targets_or_controls=False,frozen_source=str(frozen) if frozen else None)
    (output/'eligibility.json').write_text(json.dumps(identity,indent=2)+'\n')
    for key in ('Q_reference','Q_window'):frame[key]=[r[key] for r in rows]
    with np.load(corpus/'observations.npz',allow_pickle=False) as raw:
        simple=[]
        for row in rows:
            x=raw[row['row_id']].T;c=np.corrcoef(x.T)[np.triu_indices(row['M'],1)];mean=x.mean(axis=1)
            simple.append(dict(mean_abs_correlation=float(abs(c).mean()),sample_Q=float(np.abs(mean[1::2]-mean[::2]).mean())))
    for key in simple[0]:frame[key]=[s[key] for s in simple]
    results=[]
    for keys,group in frame[held & eligible].groupby(['view','M','T']):
        means=group.groupby('r')[['q','Q_reference','Q_window']].mean()
        residual=group[['q','Q_reference']]-group.groupby('r')[['q','Q_reference']].transform('mean')
        results.append(dict(view=keys[0],M=int(keys[1]),T=int(keys[2]),rows=len(group),
            rho_reference=corr(group.q,group.Q_reference),rho_window=corr(group.q,group.Q_window),
            control_mean_rho=corr(means.q,means.Q_reference),within_r_rho=corr(residual.q,residual.Q_reference),
            raw_correlation_rho=corr(group.mean_abs_correlation,group.Q_reference),sample_Q_rho=corr(group.sample_Q,group.Q_reference)))
    frame.to_csv(output/'scores.csv',index=False)
    summary=dict(status='exploratory; not independent confirmation',rows=len(frame),spis=len(order),
        selected_features=len(keep),passes_row_gate=valid_gate,geometry=geometry,results=results,
        maximum_selected_missingness=float(missing.max()),excluded_rows=int((~eligible).sum()))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    groups=list(frame[held].groupby(['view','M','T']))
    fig,axes=plt.subplots(len(groups),2,figsize=(9,3*len(groups)),squeeze=False,constrained_layout=True)
    for (keys,group),axs in zip(groups,axes):
        # Arbitrary PC sign is displayed consistently using development only.
        fit=frame[dev & eligible]
        sign=-1 if corr(fit.q,fit.Q_reference)<0 else 1
        means=group.groupby('r')[['q','Q_reference']].mean();ax=axs[0];right=ax.twinx()
        ax.plot(means.index,means.Q_reference,'ko-',label='physical Q');right.plot(means.index,sign*means.q,'s-',color='#31688e',label='q')
        ax.axvline(3.86212,color='.5',ls=':');ax.set(xlabel='r',ylabel='physical Q',title=f'{keys}')
        right.set_ylabel('q (sign oriented for display)')
        axs[1].scatter(group.Q_reference,sign*group.q,c=group.r,cmap='viridis',s=18)
        axs[1].set(xlabel='physical Q',ylabel='q')
    fig.savefig(output/'tracking.png',dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['corpus','mpi-root','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--frozen',type=Path);a=p.parse_args();run(a.corpus,a.mpi_root,a.output,a.frozen)
