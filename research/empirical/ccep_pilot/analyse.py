"""Validate complete CCEP tables and produce exploratory, participant-level geometry."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from mtsspi.corpus import ArchiveCorpus
from mtsspi.run_corpus import completion_error
from mtsspi.spi_spi_contract import build_unified_features
from research.empirical.ccep_pilot.acquire import sha256
from research.empirical.ccep_pilot.plots import style, save

EXPECTED_SOURCE='/scratch/ql44/we2614/mts-spi-study/operations/sources/pyspi-pair-sampling-fd0de71'


def correlation_endpoint(name):
    return name.startswith(('corr_','xcorr')) or name in {'cov_EmpiricalCovariance','cov-sq_EmpiricalCovariance'}


def analyse(root):
    root=Path(root);rows=[];values=[];matrices=[];names=None;coverage=[]
    for config in sorted((root/'corpora').glob('*/external.yaml')):
        corpus=ArchiveCorpus.from_file(config)
        manifest=json.loads((config.parent/'manifest.json').read_text())
        assert sha256(corpus.archive)==manifest['archive_sha256']
        for entry,row in zip(corpus.entries(),manifest['rows'],strict=True):
            error=completion_error(corpus,entry)
            if error: raise ValueError(f'{entry.name}: {error}')
            folder=entry.output_dir(corpus)
            metadata=json.loads((folder/'meta.json').read_text())
            version=metadata['pyspi']['version']
            assert version['computation']=='3.0.0.r9',version
            assert EXPECTED_SOURCE in json.dumps(version),version
            assert not metadata['pyspi'].get('sampling')
            current=[s['name'] for s in metadata['pyspi']['spis']]
            if names is None: names=current
            assert names==current and len(names)==289
            with np.load(folder/'spi_mpis.npz') as archive: table={name:archive[name] for name in names}
            edges=~np.eye(entry.M,dtype=bool)
            finite=float(np.mean([np.isfinite(a[edges]).mean() for a in table.values()]))
            if finite<.9: raise ValueError(f'{entry.name}: finite-edge coverage {finite:.3f}')
            feature=build_unified_features(table,names)
            rows.append(row);values.append(feature.z);matrices.append(table)
            coverage.append(dict(name=entry.name,finite_edge_fraction=finite,failed_spis=len(metadata['pyspi']['errors']),
                                 seconds=metadata['job']['compute_seconds'],workers=metadata['job']['n_jobs'],
                                 peak_memory_mib=metadata['job']['peak_memory_mib'],mpi_sha256=sha256(folder/'spi_mpis.npz')))
    if not rows: raise ValueError('No complete participant corpora')
    z=np.asarray(values,dtype=float)
    primary=np.array([not row['reference_control'] for row in rows])
    development=np.array([not row['reference_control'] and row['window']==0 for row in rows])
    assert development.sum()>=4
    # Completeness across primary windows; never impute failed/constant features.
    keep=np.isfinite(z[primary]).all(axis=0)
    eligible=np.flatnonzero(keep)
    keep[eligible]=z[development][:,eligible].std(axis=0)>1e-8
    center=z[development][:,keep].mean(axis=0)
    x=z[:,keep]-center
    _,singular,loadings=np.linalg.svd(x[development],full_matrices=False)
    for v in loadings:
        if v[np.argmax(abs(v))]<0:v*=-1
    scores=x@loadings[:2].T
    indices=np.flatnonzero(keep)
    schema=[feature.schema[i] for i in indices]
    ranked=np.argsort(-abs(loadings[0]),kind='stable')
    table=pd.DataFrame([dict(rank=rank+1,feature_index=int(indices[j]),spi_a=schema[j].spi_a,spi_b=schema[j].spi_b,
        weight=float(loadings[0,j]),both_correlation=correlation_endpoint(schema[j].spi_a) and correlation_endpoint(schema[j].spi_b))
        for rank,j in enumerate(ranked)])
    noncorr=np.array([not(correlation_endpoint(s.spi_a) and correlation_endpoint(s.spi_b)) for s in schema])
    _,_,v=np.linalg.svd(x[development][:,noncorr],full_matrices=False)
    reduced=x[:,noncorr]@v[0]
    sensitivity=float(abs(np.corrcoef(reduced[primary],scores[primary,0])[0,1]))
    out=root/'analysis';out.mkdir(exist_ok=True)
    table.to_csv(out/'feature-rankings.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    np.savez_compressed(out/'features.npz',z=z,selected=keep,center=center,loadings=loadings[:2],scores=scores,
                        development=development,primary=primary,spi_order=np.array(names))
    subject_rows=[]
    for subject in sorted({r['subject'] for r in rows}):
        selected=np.array([r['subject']==subject and not r['reference_control'] for r in rows])
        reference=np.array([r['subject']==subject and r['reference_control'] for r in rows])
        subject_rows.append(dict(subject=subject,pc1_first=float(scores[selected][0,0]),
            pc1_min=float(scores[selected,0].min()),pc1_max=float(scores[selected,0].max()),
            reference_pc1=float(scores[reference,0][0]) if reference.any() else None))
    pd.DataFrame(subject_rows).to_csv(out/'participant-scores.csv',index=False)
    contrasts=pd.concat([table.head(1),table.loc[~table.both_correlation].head(2)]).drop_duplicates('feature_index')
    first=np.flatnonzero(development)
    extremes=first[np.argsort(scores[first,0])[[0,-1]]]
    edges=[]
    for row_index in extremes:
        record=rows[row_index];m=record['M'];mask=~np.eye(m,dtype=bool)
        for chosen in contrasts.itertuples():
            a,b=[matrices[row_index][name] for name in (chosen.spi_a,chosen.spi_b)]
            deletion=[]
            for channel in range(m):
                held=mask.copy();held[channel,:]=False;held[:,channel]=False
                deletion.append(float(np.corrcoef(a[held],b[held])[0,1]))
            edges.append(dict(subject=record['subject'],spi_a=chosen.spi_a,spi_b=chosen.spi_b,
                z=float(np.corrcoef(a[mask],b[mask])[0,1]),mean_a=float(a[mask].mean()),mean_b=float(b[mask].mean()),
                sd_a=float(a[mask].std()),sd_b=float(b[mask].std()),delete_min=min(deletion),delete_max=max(deletion)))
    pd.DataFrame(edges).to_csv(out/'edge-contrasts.csv',index=False)
    report=dict(rows=rows,participants=int(development.sum()),primary_windows=int(primary.sum()),
        reference_controls=int((~primary).sum()),features=int(keep.sum()),
        explained_variance=(singular**2/(singular**2).sum()).tolist(),top20_double_correlation=int(table.head(20).both_correlation.sum()),
        correlation_exclusion_score_r=sensitivity,nonfinite_reference_score_rows=int((~np.isfinite(scores[~primary]).all(axis=1)).sum()),
        interpretation='Exploratory first-window PCA; other windows and acquisition-reference controls probe sensitivity. No group or evoked-response association established.',
        missingness='Primary-window completeness; reference projections with missing selected features remain NaN.',
        source=EXPECTED_SOURCE,script_sha256=sha256(__file__))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    plot_results(root,rows,scores,development,table,contrasts,extremes,matrices,report)
    return report


def plot_results(root,rows,scores,development,table,contrasts,extremes,matrices,report):
    plt=style();out=root/'figures';subjects=sorted({r['subject'] for r in rows})
    colors=dict(zip(subjects,plt.cm.viridis(np.linspace(.05,.88,len(subjects)))))
    fig,axes=plt.subplots(1,2,figsize=(6.6,3.4))
    for subject in subjects:
        primary=np.array([r['subject']==subject and not r['reference_control'] for r in rows])
        reference=np.array([r['subject']==subject and r['reference_control'] for r in rows])
        axes[0].plot(scores[primary,0],scores[primary,1],'o-',color=colors[subject],label=subject[-2:],alpha=.8)
        if reference.any():axes[0].scatter(scores[reference,0],scores[reference,1],marker='x',s=22,color=colors[subject])
        base=np.flatnonzero(primary)[0]
        axes[0].scatter(*scores[base,:2],s=35,facecolors='none',edgecolors=colors[subject])
    axes[0].set(xlabel='PC1',ylabel='PC2',title='First-window fit; repeated-window paths')
    axes[0].legend(title='Participant',fontsize=6,title_fontsize=7)
    axes[1].bar(np.arange(1,len(report['explained_variance'])+1),np.asarray(report['explained_variance'])*100,color='#26828e')
    axes[1].set(xlabel='Component',ylabel='Explained variance (%)',title='Five independent participants')
    for ax in axes:ax.set_box_aspect(1)
    save(fig,out,'pca')
    galleries=[table.head(8),table.loc[~table.both_correlation].head(8)]
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,chosen,title in zip(axes,galleries,['Largest PC1 weights','Exclude correlation–correlation pairs']):
        labels=[f'{r.spi_a}\n× {r.spi_b}' for r in chosen.itertuples()]
        ax.barh(np.arange(len(chosen)),chosen.weight,color='#26828e');ax.set_yticks(np.arange(len(chosen)),labels,fontsize=5.5)
        ax.invert_yaxis();ax.axvline(0,color='.5',lw=.5);ax.set(xlabel='PC1 loading',title=title)
    save(fig,out,'loadings')
    fig,axes=plt.subplots(2,len(contrasts),figsize=(3.2*len(contrasts),6.3),squeeze=False)
    for column,chosen in enumerate(contrasts.itertuples()):
        pairs=[]
        for index in extremes:
            mask=~np.eye(rows[index]['M'],dtype=bool)
            pairs.append([matrices[index][name][mask] for name in (chosen.spi_a,chosen.spi_b)])
        limits=[]
        for endpoint in (0,1):
            vals=np.concatenate([p[endpoint] for p in pairs]);pad=max(float(np.ptp(vals))*.04,1e-8)
            limits.append((vals.min()-pad,vals.max()+pad))
        for row_index,(index,pair) in enumerate(zip(extremes,pairs)):
            ax=axes[row_index,column];subject=rows[index]['subject']
            ax.scatter(*pair,s=3,alpha=.2,color=colors[subject],rasterized=True,linewidths=0)
            ax.set(xlim=limits[0],ylim=limits[1],title=f'{subject[-2:]} · z={np.corrcoef(*pair)[0,1]:.3f}')
            ax.set_xlabel(chosen.spi_a,fontsize=6);ax.set_ylabel(chosen.spi_b,fontsize=6);ax.set_box_aspect(1)
    save(fig,out,'edge-contrasts')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(analyse(parser.parse_args().root),indent=2))
