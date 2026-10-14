"""Interpret verified pilot outputs; preserve the original PCA and frozen inputs."""
from __future__ import annotations
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
from .acquire import sha256
from .analyse import correlation_endpoint
from .plots import style, save
from .prepare import lag_matrix


def basic_distance_endpoint(name):
    return correlation_endpoint(name) or name.startswith(('pdist_', 'xpdist_', 'bary_', 'bary-'))


def agreement(a, b):
    edge=np.triu_indices(len(a),1)
    return float(np.corrcoef(a[edge],b[edge])[0,1])


def circular_lag(x, lag, fs=128):
    """Exact circular symmetric lag covariance via the real cross-spectrum."""
    x=x-x.mean(axis=1,keepdims=True)
    x=x/x.std(axis=1,keepdims=True)
    n=x.shape[1];f=np.fft.rfftfreq(n,1/fs);fourier=np.fft.rfft(x)
    weight=np.full(len(f),2.);weight[0]=1
    if n%2==0:weight[-1]=1
    covariance=((fourier*(weight*np.cos(2*np.pi*f*lag/fs)))@fourier.conj().T).real/n**2
    negative=(f>8)&(f<24)  # Negative cosine lobe for tau=4 at128Hz.
    power=(abs(fourier)**2*weight).sum(axis=0)
    removed=np.fft.irfft(fourier*(~negative),n=n)
    return covariance, float(power[negative].sum()/power.sum()), removed


def delete_range(a,b):
    values=[]
    for i in range(len(a)):
        kept=np.arange(len(a))!=i
        values.append(agreement(a[np.ix_(kept,kept)],b[np.ix_(kept,kept)]))
    return min(values),max(values)


def shared_count(vectors):
    return int(np.min(np.asarray(vectors),axis=0).sum())


def coverage_gate(root, out):
    """Coverage only: never use latency magnitudes to select contacts or sites."""
    responses=pd.read_csv(root/'responses/matched-latencies.csv')
    assert not responses.duplicated(['subject','site','response']).any()
    detected=responses[responses.detected];sites={};strata=set();provenance={}
    for who,group in detected.groupby('subject'):
        path=next((root/'metadata'/who).glob('**/*electrodes.tsv'))
        provenance[str(path.relative_to(root))]=sha256(path)
        electrodes=pd.read_csv(path,sep='\t').set_index('name');local=[]
        for site,receivers in group.groupby('site'):
            if len(receivers)<8:continue
            stim=site.split('-')[:2]
            centre=electrodes.loc[stim,['x','y','z']].to_numpy(float).mean(axis=0)
            counts=Counter()
            for response in receivers.response:
                e=electrodes.loc[response];xyz=e[['x','y','z']].to_numpy(float)
                if not np.isfinite(xyz).all() or pd.isna(e.Destrieux_label_text):continue
                counts[(e.Destrieux_label_text,int(np.linalg.norm(xyz-centre)//20))]+=1
            strata.update(counts);local.append((site,counts))
        sites[who]=local
    keys=sorted(strata)
    vectors={who:[(site,np.array([counts[k] for k in keys])) for site,counts in rows]
             for who,rows in sites.items()}
    rows=[]
    for subset in itertools.combinations(sorted(vectors),4):
        best=-1;chosen=None
        for combo in itertools.product(*(vectors[who] for who in subset)):
            count=shared_count([v for _,v in combo])
            if count>best:best=count;chosen=[site for site,_ in combo]
        rows.append(dict(participants=','.join(w[-2:] for w in subset),maximum_common_receivers=max(0,best),sites=chosen))
    report=dict(site_counts={w:len(rows) for w,rows in sites.items()},comparisons=rows,
                required_receivers=8,passed=any(r['maximum_common_receivers']>=8 for r in rows),
                matching='Exact receiver Destrieux label × 20mm stimulation-centre distance bin; one site per participant. Upper bound before receiver-pair distance matching.',
                scope='Pilot feasibility only; failure under this matching does not rule out coarser anatomical matching or another cohort.',
                metadata_sha256=provenance,response_sha256=sha256(root/'responses/matched-latencies.csv'))
    (out/'coverage-gate.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def interpret(root):
    root=Path(root);out=root/'interpretation';out.mkdir(exist_ok=True)
    report=json.loads((root/'analysis/summary.json').read_text());rows=report['rows']
    selection=json.loads((Path(__file__).parent/'configs/analysis-selection.json').read_text())
    coverage=pd.read_csv(root/'analysis/coverage.csv').set_index('name')
    series={};tables={};input_hashes={}
    for name in selection['corpora']:
        folder=root/'corpora'/name;manifest=json.loads((folder/'manifest.json').read_text())
        archive=folder/'observations.npz';assert sha256(archive)==manifest['archive_sha256']
        input_hashes[name]=manifest['archive_sha256']
        with np.load(archive) as bank:
            for row in manifest['rows']:series[row['name']]=bank[row['name']]
        for path in folder.glob('**/meta.json'):
            meta=json.loads(path.read_text());name=meta['source']['member'];mpi=path.parent/'spi_mpis.npz'
            assert sha256(mpi)==coverage.loc[name,'mpi_sha256']
            with np.load(mpi) as bank:tables[name]={k:bank[k] for k in bank.files}
    assert set(series)==set(tables)=={r['name'] for r in rows} and len(rows)==19
    ranks=pd.read_csv(root/'analysis/feature-rankings.csv')
    ranks['both_correlation']=[correlation_endpoint(r.spi_a) and correlation_endpoint(r.spi_b) for r in ranks.itertuples()]
    ranks['both_basic_distance']=[basic_distance_endpoint(r.spi_a) and basic_distance_endpoint(r.spi_b) for r in ranks.itertuples()]
    ranks.to_csv(out/'feature-rankings.csv',index=False)
    with np.load(root/'analysis/features.npz') as bank:
        z=bank['z'];indices=np.flatnonzero(bank['selected']);x=z[:,indices]-bank['center']
        development=bank['development'];primary=bank['primary'];scores=bank['scores']
    sensitivity={}
    for label in ['both_correlation','both_basic_distance']:
        excluded=set(ranks.loc[ranks[label],'feature_index']);keep=np.array([i not in excluded for i in indices])
        _,_,v=np.linalg.svd(x[development][:,keep],full_matrices=False)
        reduced=x[:,keep]@v[0]
        sensitivity[label]=dict(retained=int(keep.sum()),score_r=float(abs(np.corrcoef(reduced[primary],scores[primary,0])[0,1])))
    features=[]
    for i,row in enumerate(rows):
        x=series[row['name']];table=tables[row['name']];edge=np.triu_indices(len(x),1)
        zero=np.corrcoef(x);lag=lag_matrix(x,4/128,128)
        np.testing.assert_allclose(zero[edge],table['cov_EmpiricalCovariance'][edge],atol=1e-12)
        np.testing.assert_allclose(lag[edge],table['corr_pearson_tau-4'][edge],atol=1e-12)
        replay,fraction,removed=circular_lag(x,4)
        lo,hi=delete_range(zero,lag)
        features.append(dict(subject=row['subject'],window=row['window'],reference_control=row['reference_control'],
            pc1=float(scores[i,0]),z_zero_lag=agreement(zero,lag),z_replay=agreement(zero,replay),
            replay_edge_rmse=float(np.sqrt(np.mean((lag[edge]-replay[edge])**2))),
            band_8_24_fraction=fraction,z_without_8_24=agreement(np.corrcoef(removed),lag_matrix(removed,4/128,128)),
            delete_min=lo,delete_max=hi,
            z_leading=agreement(table['corr_spearman_tau-4'],table['pdist_braycurtis']),
            bray_vs_zero=agreement(table['pdist_braycurtis'],zero)))
    features=pd.DataFrame(features);features.to_csv(out/'character-axis.csv',index=False)
    gate=coverage_gate(root,out)
    summary=dict(first_window_pc1_variance=report['explained_variance'][0],
                 top20_correlation_pairs=int(ranks.head(20).both_correlation.sum()),
                 top20_basic_distance_pairs=int(ranks.head(20).both_basic_distance.sum()),
                 sensitivity=sensitivity,max_replay_edge_rmse=float(features.replay_edge_rmse.max()),
                 max_replay_z_error=float(abs(features.z_zero_lag-features.z_replay).max()),
                 coverage_gate_passed=gate['passed'],n1_association_run=False,
                 interpretation='Exploratory lag-profile/spectral organisation. No conduction-delay, nonlinear-mechanism or baseline–N1 association claim.',
                 inputs=input_hashes,analysis_summary_sha256=sha256(root/'analysis/summary.json'),
                 correlation_filter_source_sha256=sha256(Path(__file__).with_name('analyse.py')),script_sha256=sha256(__file__))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plots(root,rows,series,tables,ranks,features,gate)
    return summary


def short_name(name):
    if name=='cov_EmpiricalCovariance':return 'Pearson (0 ms)'
    return name.replace('corr_','').replace('_tau-4',' 31ms').replace('cov_','cov ').replace('pdist_','distance ').replace('EmpiricalCovariance','Pearson').replace('_',' ')


def plots(root,rows,series,tables,ranks,features,gate):
    import seaborn as sns
    plt=style();folder=root/'figures'
    fig,axes=plt.subplots(3,2,figsize=(7.6,8.8));panels=[]
    for ax,subject in zip(axes.flat,sorted({r['subject'] for r in rows})):
        row=next(r for r in rows if r['subject']==subject and not r['reference_control'])
        x=series[row['name']];q25,q75=np.quantile(x,[.25,.75],axis=1)
        scaled=(x-np.median(x,axis=1,keepdims=True))/np.maximum(q75-q25,1e-12)[:,None]
        im=ax.imshow(scaled,origin='lower',aspect='auto',extent=(0,32,.5,len(x)+.5),vmin=-3,vmax=3,
                     cmap=sns.color_palette('icefire',as_cmap=True),interpolation='nearest')
        ax.set(title=f"{subject[-2:]} · retained w{row['window']} · M={len(x)}",xlabel='Window time (s)',ylabel='Channel');ax.set_box_aspect(1);panels.append(ax)
    axes.flat[-1].set_visible(False)
    fig.colorbar(im,ax=panels,shrink=.35,pad=.02,label='Channel median/IQR; display ±3')
    save(fig,folder,'pilot-retained-heatmaps')
    fig,axes=plt.subplots(1,2,figsize=(10,4.1))
    for ax,chosen,title in zip(axes,[ranks.head(6),ranks[~ranks.both_basic_distance].head(6)],['Largest PC1 weights','Beyond correlation/distance pairs']):
        labels=[short_name(r.spi_a)+'\n× '+short_name(r.spi_b) for r in chosen.itertuples()]
        ax.barh(np.arange(len(chosen)),chosen.weight,color='#26828e');ax.set_yticks(np.arange(len(chosen)),labels,fontsize=6)
        ax.invert_yaxis();ax.set(xlabel='PC1 loading',title=title);ax.axvline(0,color='.5',lw=.5)
    save(fig,folder,'interpreted-loadings')
    fig,axes=plt.subplots(2,2,figsize=(6.8,6.6))
    for col,who in enumerate(['02','10']):
        row=next(r for r in rows if r['subject'].endswith(who) and not r['reference_control'] and r['window']==0)
        table=tables[row['name']];edge=np.triu_indices(row['M'],1)
        for ax,a,b in [(axes[0,col],'corr_spearman_tau-4','pdist_braycurtis'),(axes[1,col],'cov_EmpiricalCovariance','corr_pearson_tau-4')]:
            ax.scatter(table[a][edge],table[b][edge],s=3,alpha=.22,color='#26828e',rasterized=True)
            ax.set(xlabel=short_name(a),ylabel=short_name(b),title=f'{who} · z={agreement(table[a],table[b]):.3f}');ax.set_box_aspect(1)
    for a,b in zip(axes[:,0],axes[:,1]):
        for getter,setter in [('get_xlim','set_xlim'),('get_ylim','set_ylim')]:
            lims=[getattr(ax,getter)() for ax in (a,b)];limit=(min(x[0] for x in lims),max(x[1] for x in lims))
            for ax in (a,b):getattr(ax,setter)(limit)
    save(fig,folder,'interpreted-edges')
    fig,axes=plt.subplots(1,3,figsize=(9.6,3.4));colors=['#26828e','#b56b22']
    for who,color in zip(['02','10'],colors):
        row=next(r for r in rows if r['subject'].endswith(who) and not r['reference_control'] and r['window']==0)
        f,p=welch(series[row['name']],fs=128,nperseg=512,axis=1);mask=(f>=.5)&(f<=45)
        axes[0].semilogy(f[mask],p.mean(axis=0)[mask],label=who,color=color)
    axes[0].axvspan(8,24,color='.8',alpha=.4);axes[0].set(xlabel='Frequency (Hz)',ylabel='Mean PSD (standardised voltage²/Hz)',title='Spectral explanation: 8–24 Hz shaded');axes[0].legend(title='Participant')
    for j,(subject,group) in enumerate(features.groupby('subject')):
        car=group[~group.reference_control];ref=group[group.reference_control]
        axes[1].scatter(np.full(len(car),j),car.z_zero_lag,s=20,color='#26828e')
        axes[1].scatter(np.full(len(ref),j),ref.z_zero_lag,marker='x',s=30,color='#b56b22')
    axes[1].set(xticks=range(5),xticklabels=['02','10','12','61','73'],xlabel='Participant',ylabel='Zero-lag × 31.25ms z',ylim=(-1.05,1.05),title='CAR windows ● · acquisition ×')
    axes[2].scatter(features.z_zero_lag,features.z_replay,s=16,color='#26828e');axes[2].plot([-1,1],[-1,1],color='.6',ls=':',lw=.8)
    axes[2].set(xlabel='Observed z',ylabel='Cross-spectrum reconstructed z',title='All 19 windows/reference controls')
    for ax in axes:ax.set_box_aspect(1)
    save(fig,folder,'spectral-character')
    fig,axes=plt.subplots(1,2,figsize=(6.7,3.5))
    axes[0].bar([s[-2:] for s in gate['site_counts']],list(gate['site_counts'].values()),color='#26828e')
    axes[0].set(xlabel='Participant',ylabel='Sites with ≥8 detected receivers',title='Uneven response coverage')
    labels=[r['participants'].replace(',','/') for r in gate['comparisons']]
    axes[1].bar(range(len(labels)),[r['maximum_common_receivers'] for r in gate['comparisons']],color='#26828e')
    axes[1].axhline(8,color='.5',ls=':',lw=.8);axes[1].set(xticks=range(len(labels)),xticklabels=labels,ylim=(0,9),xlabel='Four-participant subset',ylabel='Maximum matched receivers / site',title='Anatomy × distance overlap')
    axes[1].tick_params(axis='x',labelrotation=45,labelsize=6)
    for ax in axes:ax.set_box_aspect(1)
    save(fig,folder,'n1-coverage-gate')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    print(json.dumps(interpret(parser.parse_args().root),indent=2))
