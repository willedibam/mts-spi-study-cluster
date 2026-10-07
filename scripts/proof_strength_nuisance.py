"""Proof classes observed at controlled coupling strength: is the mean-SPI baseline strength, and SPI-SPI character?

Eight coupled classes (two VARs, Kuramoto, wave, four CML regimes) and two independent-noise references, M=16, T=1000.
Every arm observes the SAME dynamical realization per (class, instance) through white Gaussian sensor noise of SD eta,
in signal-SD units; arms differ only in how eta is set per recording:
  native     eta=0.
  mild       eta~U[.3,.9], independent of class: recording quality varies (the construction that held for 2:1 locking).
  wide       Pearson attenuation u=1/(1+eta^2)~U[.25,.95], independent of class.
  equalised  eta solves mean|r|=.14: strength normalised to one value in every class.
  matched    eta solves mean|r|=s, s~U[.08,.20]: strength normalised to one DISTRIBUTION in every class and randomised.
PROOF_STRENGTH_SIZE=m24 selects a fresh-seed repeat of native and matched at M=24 (added after the M=16 outcome was read).
PROOF_STRENGTH_SIZE=uncoupled builds each recording of a coupled class from 16 independent realizations, one channel from each:
every channel keeps its class's own dynamics and no channel has interacted with another (added 7 October).
Sensor noise only attenuates, so a recording natively weaker than its target is left as generated (capped; recorded).
Gaussian and Cauchy noise have no coupling to attenuate (same-family observation noise leaves their law unchanged);
one set is generated and joins every arm.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import io
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
RUNS={'m16':dict(run='proof-strength-nuisance-261006',M=16,seed=261101,arms=('native','mild','wide','equalised','matched')),
      'm24':dict(run='proof-strength-nuisance-m24-261006',M=24,seed=261111,arms=('native','matched')),
      'uncoupled':dict(run='proof-uncoupled-261007',M=16,seed=261101,arms=('uncoupled',))}
SIZE=os.environ.get('PROOF_STRENGTH_SIZE','m16')
RUN,M,SEED,ARMS=(RUNS[SIZE][k] for k in ('run','M','seed','arms'))
DATA=ROOT/'data/proof'/RUN
OUT=ROOT/'results/proof'/RUN
REMOTE='/scratch/ql44/we2614/mts-spi-study/proof/'+RUN
T,INSTANCES=1000,40
PYSPI_SEED=SEED+1
EQUAL,TARGET=.14,(.08,.20)
CML=dict(transients=2000,sample_every=1,zscore=False)
# VARs are literal ring coefficients (self, per-neighbour) at spectral radius .98; Kuramoto is the proof's fast-frequency
# all-to-all system at K=3.5 (partial synchrony) instead of K=-4, so its native strength exceeds the target range.
CLASSES={
    'var-self-0.2':('varma',dict(phi=.2,coupling=.39,ma_phi=0.,ma_coupling=0.,noise_std=.1,topology='ring-symmetric',transients=2000,zscore=False)),
    'var-self-0.7':('varma',dict(phi=.7,coupling=.14,ma_phi=0.,ma_coupling=0.,noise_std=.1,topology='ring-symmetric',transients=2000,zscore=False)),
    'kuramoto':('kuramoto',dict(K=3.5,dt=.00625,omega_mean=3.,omega_std=1.73205,eta=0.,output='sin',connectivity='all-to-all',transients=2000,zscore=False)),
    'wave-1d':('wave_1d',dict(c=10.,n_modes=5,ic_decay=1.5,noise_std=.01,zscore=False)),
    'frozen-chaos':('cml_logistic',dict(alpha=1.45,eps=.2,**CML)),
    'sti-i':('cml_logistic',dict(alpha=1.7522,eps=.00115,**CML)),
    'defect-turbulence':('cml_logistic',dict(alpha=1.895,eps=.1,**CML)),
    'fdstc':('cml_logistic',dict(alpha=2.,eps=.3,**CML)),
    'gaussian-noise':('gaussian_noise',dict(zscore=False)),
    'cauchy-noise':('cauchy_noise',dict(zscore=False)),
}
NOISE=('gaussian-noise','cauchy-noise')
CML_PANEL=('frozen-chaos','sti-i','defect-turbulence','fdstc')
INTER_PANEL=('var-self-0.2','var-self-0.7','kuramoto','wave-1d','defect-turbulence','gaussian-noise','cauchy-noise')
OFF=~np.eye(M,dtype=bool)


def strength(x):
    return float(abs(np.corrcoef(x)[OFF]).mean())


def dynamics(name,instance):
    """One (M,T) realization, channels z-scored; shared by every arm."""
    from src import generators
    generator,params=CLASSES[name]
    rng=np.random.default_rng(np.random.SeedSequence([SEED,list(CLASSES).index(name),instance]))
    with contextlib.redirect_stdout(io.StringIO()):
        x=np.asarray(getattr(generators,'generate_'+generator)(M=M,T=T,rng=rng,**params),float).T
    assert x.shape==(M,T) and np.isfinite(x).all() and x.std(1).min()>0
    return (x-x.mean(1,keepdims=True))/x.std(1,keepdims=True)


def observe(x,arm,rng):
    """Sensor noise at the arm's level; returns the z-scored observation and its provenance."""
    native=strength(x);noise=rng.normal(size=x.shape);level=float(rng.uniform());target=None
    if arm=='native':eta=0.
    elif arm=='mild':eta=.3+.6*level
    elif arm=='wide':eta=float(np.sqrt(1/(.25+.7*level)-1))
    else:
        target=EQUAL if arm=='equalised' else TARGET[0]+(TARGET[1]-TARGET[0])*level
        eta=0. if native<=target else float(brentq(lambda e:strength(x+e*noise)-target,0,200,xtol=1e-10))
    y=x+eta*noise;y=(y-y.mean(1,keepdims=True))/y.std(1,keepdims=True)
    return y,dict(eta=eta,target=target,capped=bool(target is not None and native<=target),native_abs_r=native,mean_abs_r=strength(y),
                  mean_abs_spearman=float(abs(spearmanr(y.T).statistic[OFF]).mean()))


def record(task):
    arm,name,instance=task
    if arm=='uncoupled':   # realizations 1000+ are disjoint from every coupled recording and from each other
        x=np.stack([dynamics(name,1000+M*instance+c)[c] for c in range(M)])
        return x,dict(eta=0.,target=None,capped=False,native_abs_r=strength(x),mean_abs_r=strength(x),mean_abs_spearman=float(abs(spearmanr(x.T).statistic[OFF]).mean()))
    x=dynamics(name,instance)
    if name in NOISE:return x,dict(eta=0.,target=None,capped=False,native_abs_r=strength(x),mean_abs_r=strength(x),
                                   mean_abs_spearman=float(abs(spearmanr(x.T).statistic[OFF]).mean()))
    return observe(x,arm,np.random.default_rng(np.random.SeedSequence([SEED+1,RUNS['m16']['arms'].index(arm),list(CLASSES).index(name),instance])))


def tasks():
    coupled=[(arm,name,i) for arm in ARMS for name in CLASSES if name not in NOISE for i in range(INSTANCES)]
    return coupled+[('all',name,i) for name in NOISE for i in range(INSTANCES) if SIZE!='uncoupled']


def prepare(workers):
    import yaml
    from scripts.cross_frequency_locking import partitions
    from scripts.spi_baseline_exploration import sha
    if (DATA/'manifest.json').exists():raise FileExistsError(DATA)
    DATA.mkdir(parents=True,exist_ok=True);rows=[];raw={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (arm,name,instance),(x,truth) in zip(tasks(),pool.map(record,tasks(),chunksize=8)):
            row=f'{arm}-{name}-i{instance:02d}';raw[row]=x
            rows.append(dict(row_id=row,corpus_index=len(rows),M=M,T=T,seed=instance,instance=instance,block=instance,label=name,system=arm,**truth))
    np.savez_compressed(DATA/'observations.npz',**raw,__dataset_names__=np.array(list(raw)),
        __labels_json__=np.array([json.dumps([r['label']]) for r in rows]),__shapes__=np.array([[M,T]]*len(rows)),
        __axis_order__=np.array(['process','observation']))
    manifest=dict(rows=rows,corpus=RUN,archive_sha256=sha(DATA/'observations.npz'),generator_sha256=sha(__file__),
        analysis_scope='Exploratory. Arms share each dynamical realization; noise classes join every arm. See the module docstring.')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config=dict(name=RUN,source=dict(format='named-npz-v1',archive=REMOTE+'/observations.npz',sha256=manifest['archive_sha256'],
        axis_order=['process','observation']),base_output_dir=REMOTE+'/mpis',pyspi_config='configs/pyspi/benchmarked_p90.yaml',
        normalise=False,random_seed=PYSPI_SEED)
    (DATA/'corpus.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    for name,indices in partitions(len(rows)).items():
        (DATA/f'{name}-indices.txt').write_text(''.join(f'{i}\n' for i in indices))
    frame=pd.DataFrame(rows)
    print(frame.groupby(['system','label'])[['native_abs_r','mean_abs_r','mean_abs_spearman','eta','capped']].agg(['mean','min','max']).round(3).to_string())
    print(len(rows),manifest['archive_sha256'])


def extract(data=DATA,out=OUT):
    """Per-SPI means and the proof notebook's symmetrised SPI-SPI block from the retrieved MPIs."""
    from scripts.spi_baseline_exploration import sha,summarize
    from src.spi_spi_contract import _similarity_matrix,_upper_symmetrized
    from src.utils import slugify
    manifest=json.loads((data/'manifest.json').read_text());mean,spread,z,seconds=[],[],[],[];order=None
    config=sha(ROOT/'configs/pyspi/benchmarked_p90.yaml')
    for r in manifest['rows']:
        folder=data/'mpis'/RUN/f"{r['corpus_index']+1:04d}-{slugify(r['row_id'],'dataset')}";meta=json.loads((folder/'meta.json').read_text())
        assert meta['status']=='complete' and meta['dataset_name']==r['row_id'] and (meta['M'],meta['T'])==(M,T)
        assert meta['source']['archive_sha256']==manifest['archive_sha256'] and meta['pyspi']['config_sha256']==config
        names=[x['name'] for x in meta['pyspi']['spis']];order=order or names;assert order==names and len(names)==289
        with np.load(folder/'spi_mpis.npz') as a:mpis={k:a[k] for k in order}
        marginal=summarize(mpis,order,symmetric=True);mean.append(marginal[:,0]);spread.append(marginal[:,1])
        with np.errstate(all='ignore'):c=_similarity_matrix(np.vstack([_upper_symmetrized(mpis[k]) for k in order]),'pearson')
        z.append(c[np.triu_indices(289,1)].astype(np.float32));seconds.append(meta['job']['compute_seconds'])
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'features.npz',mean=np.array(mean),spread=np.array(spread),z=np.array(z),row_id=np.array([r['row_id'] for r in manifest['rows']]),
        spi_order=np.array(order),compute_seconds=np.array(seconds))
    print(len(z),'records; compute seconds median/max',np.median(seconds).round(),np.max(seconds).round(),'; finite z fraction',np.isfinite(z).mean().round(4))


# ---- analysis: every fit is unsupervised and separate per arm and panel, as in the proof notebook ----
PANELS={'inter':INTER_PANEL,'cml':CML_PANEL}
REPRESENTATIONS={'corr':'Correlation-family means','mean':'Mean of each SPI, $m$','z':'SPI-SPI, $z$'}
CORRELATION=('cov','cov-sq','spearmanr','spearmanr-sq','kendalltau','kendalltau-sq','xcorr','xcorr-sq')
DISPLAY={'var-self-0.2':'VAR: self .2, neighbour .39','var-self-0.7':'VAR: self .7, neighbour .14','kuramoto':'Kuramoto','wave-1d':'Wave equation',
         'frozen-chaos':'Frozen chaos','sti-i':'Spatiotemporal intermittency','defect-turbulence':'Defect turbulence','fdstc':'Fully developed turbulence',
         'gaussian-noise':'Gaussian noise','cauchy-noise':'Cauchy noise'}
COLORS={'frozen-chaos':'#CC79A7','sti-i':'#E69F00','defect-turbulence':'#009E73','fdstc':'#D55E00','var-self-0.2':'#0072B2','var-self-0.7':'#56B4E9',
        'kuramoto':'#332288','wave-1d':'#882255','gaussian-noise':'#555555','cauchy-noise':'#999933'}
UMAP_SETTINGS=dict(n_neighbors=25,min_dist=.25,metric='euclidean',random_state=260924)


def load(size=SIZE):
    data,out=(ROOT/kind/'proof'/RUNS[size]['run'] for kind in ('data','results'))
    rows=pd.DataFrame(json.loads((data/'manifest.json').read_text())['rows']);bank=dict(np.load(out/'features.npz'))
    np.testing.assert_array_equal(bank['row_id'],rows.row_id)
    family=np.array([str(s).split('_')[0] for s in bank['spi_order']]);bank['corr']=bank['mean'][:,np.isin(family,CORRELATION)]
    return rows,bank


def arms_of(rows):
    return [a for a in dict.fromkeys(rows.system) if a!='all']


def select(rows,arm,panel):
    """Rows of one panel in one arm, or in all arms pooled; the unmodulated noise references join every arm."""
    arms=arms_of(rows) if arm=='pooled' else [arm]
    return np.flatnonzero(rows.label.isin(PANELS[panel]).to_numpy()&rows.system.isin(arms+['all']).to_numpy())


def prepare_features(x,standard):
    """Proof preprocessing: features finite in >=95% of rows, median imputation, drop near-constant, centre.
    Means are also scaled to unit variance and clipped at five SD (their units differ); z is only centred."""
    x=np.asarray(x,float);x=x[:,np.isfinite(x).mean(0)>=.95];x=np.where(np.isfinite(x),x,np.nanmedian(np.where(np.isfinite(x),x,np.nan),axis=0))
    x=x[:,x.std(0)>1e-8];x=x-x.mean(0)
    return np.clip(x/x.std(0),-5,5) if standard else x


def embed(bank,index,key,umap=True):
    from sklearn.decomposition import PCA
    x=prepare_features(bank[key][index],key!='z');pca=PCA(min(50,*x.shape),svd_solver='full').fit(x);scores=pca.transform(x)
    result=dict(scores=scores,evr=pca.explained_variance_ratio_,features=x.shape[1])
    if umap:
        from umap import UMAP
        result['umap']=UMAP(**UMAP_SETTINGS,n_jobs=1).fit_transform(scores)
    return result


def geometry(scores,labels,strength_,coupled,instance):
    """Label-free geometry scored against class afterwards, plus a supervised ceiling.
    purity: leave-one-out 5-nearest-neighbour class agreement in PCA space; silhouette: by class;
    rho_strength: |Spearman| of PC1 with observed mean |r| over coupled rows; ceiling: grouped five-fold logistic accuracy on ten PCs."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import GroupKFold,cross_val_predict
    from sklearn.neighbors import NearestNeighbors
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    near=NearestNeighbors(n_neighbors=6).fit(scores).kneighbors(scores,return_distance=False)[:,1:]
    purity=float(np.mean([np.mean(labels[n]==labels[i]) for i,n in enumerate(near)]))
    predicted=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000)),scores[:,:10],labels,groups=instance,cv=GroupKFold(5))
    rho=abs(spearmanr(scores[coupled,0],strength_[coupled]).statistic) if np.ptp(strength_[coupled])>1e-6 else np.nan
    return dict(purity=purity,silhouette=float(silhouette_score(scores,labels)),rho_strength=float(rho),ceiling=float(np.mean(predicted==labels)))


def metrics(rows,bank):
    out=[]
    for panel in PANELS:
        for arm in arms_of(rows)+['pooled']:
            index=select(rows,arm,panel);part=rows.iloc[index];y=part.label.to_numpy();s=part.mean_abs_r.to_numpy();c=~part.label.isin(NOISE).to_numpy()
            common=dict(panel=panel,arm=arm,n=len(index),chance=1/part.label.nunique())
            out.append(dict(**common,representation='strength',features=1,pc1=1.,**geometry(s[:,None],y,s,c,part.instance.to_numpy())))
            for key in REPRESENTATIONS:
                e=embed(bank,index,key,umap=False)
                out.append(dict(**common,representation=key,features=e['features'],pc1=float(e['evr'][0]),**geometry(e['scores'],y,s,c,part.instance.to_numpy())))
                # The obvious repair of the mean baseline: discard its leading component, which the nuisance arms give to strength.
                if key=='mean':out.append(dict(**common,representation='mean, PC1 removed',features=e['features'],pc1=float(e['evr'][1]),
                    **geometry(e['scores'][:,1:],y,s,c,part.instance.to_numpy())))
    return pd.DataFrame(out)


def silhouette_gap(rows,bank,arm,panel,draws=2000,seed=0):
    """Silhouette of z minus that of the mean vector, with a 95% interval from resampling instances (embeddings held fixed)."""
    from sklearn.metrics import silhouette_samples
    index=select(rows,arm,panel);part=rows.iloc[index];y=part.label.to_numpy();instance=part.instance.to_numpy()
    d=silhouette_samples(embed(bank,index,'z',umap=False)['scores'],y)-silhouette_samples(embed(bank,index,'mean',umap=False)['scores'],y)
    per=np.array([d[instance==i].mean() for i in np.unique(instance)]);rng=np.random.default_rng(seed)
    boot=per[rng.integers(0,len(per),(draws,len(per)))].mean(1)
    return float(d.mean()),*np.quantile(boot,[.025,.975]).round(3)


def sensitivity(rows,bank,panel):
    """Class silhouette in 50 PCs under other scalings of the mean vector and of z; rows are scalings, columns are arms."""
    from scipy.stats import norm,rankdata
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    def finite(x):
        x=np.asarray(x,float);x=x[:,np.isfinite(x).mean(0)>=.95];x=np.where(np.isfinite(x),x,np.nanmedian(np.where(np.isfinite(x),x,np.nan),axis=0))
        return x[:,x.std(0)>1e-8]
    def robust(x):
        low,mid,high=np.quantile(x,[.25,.5,.75],axis=0);keep=high-low>1e-12
        return np.clip((x[:,keep]-mid[keep])/(high-low)[keep],-5,5)
    scalings={'m, unit variance (shown)':lambda b:prepare_features(b['mean'],True),'m, interquartile range':lambda b:robust(finite(b['mean'])),
        'm, rank-Gaussian':lambda b:norm.ppf((np.apply_along_axis(rankdata,0,finite(b['mean']))-.5)/len(b['mean'])),
        'z, centred (shown)':lambda b:prepare_features(b['z'],False),'z, unit variance':lambda b:prepare_features(b['z'],True)}
    out={}
    for arm in [a for a in ORDER if a in arms_of(rows)]:
        index=select(rows,arm,panel);y=rows.label.to_numpy()[index];part={k:bank[k][index] for k in ('mean','z')}
        out[arm]={name:silhouette_score(PCA(50,svd_solver='full').fit_transform(f(part)),y) for name,f in scalings.items()}
    return pd.DataFrame(out).round(2)


ORDER=('native','equalised','mild','matched','wide','pooled')   # display order: strength fixed, then increasingly varied
CURVES={'strength':('Mean $|r|$ alone','#009E73','v'),'corr':('Correlation-family means','#E69F00','s'),'mean':('Mean of each SPI, $m$','#D55E00','o'),
        'mean, PC1 removed':('$m$ without its PC1','#D55E00','x'),'z':('SPI-SPI, $z$','#0072B2','D')}


def table(m,panel,column):
    """One metric of one panel: rows are representations, columns are arms."""
    arms=[a for a in ORDER if a in set(m.arm)]
    return m[m.panel.eq(panel)].pivot(index='representation',columns='arm',values=column).loc[list(CURVES),arms].round(2)


def metrics_figure(m,out):
    """Cluster compactness and neighbour agreement of each representation across arms."""
    plt=style();arms=[a for a in ORDER if a in set(m.arm)];columns=[('silhouette','Silhouette by class'),('purity','5-neighbour class agreement')]
    fig,axes=plt.subplots(2,2,figsize=(7.6,5.8),layout='constrained',sharex=True)
    for row,panel in zip(axes,PANELS):
        for ax,(column,label) in zip(row,columns):
            for key,(name,color,marker) in CURVES.items():
                ax.plot(range(len(arms)),table(m,panel,column).loc[key],marker=marker,ms=4.5,lw=1.2,ls='--' if 'removed' in key else '-',color=color,label=name)
            if column=='purity':ax.axhline(m[m.panel.eq(panel)].chance.iloc[0],color='.6',lw=.8,ls=':')
            ax.set(xticks=range(len(arms)),ylabel=label);ax.set_xticklabels(arms,rotation=30,ha='right')
        row[0].set_title({'inter':'Across classes','cml':'Within CML'}[panel],loc='left')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncols=3,fontsize=8)
    return save(fig,out,'metrics')


def strength_dependence(rows,bank,arm,names):
    """Per feature: |Spearman| with observed mean |r| among recordings of one class, averaged over the classes."""
    from scipy.stats import rankdata
    index=np.flatnonzero(rows.system.eq(arm).to_numpy()&rows.label.isin(names).to_numpy());y=rows.label.to_numpy()[index];out={}
    rank=lambda v:(lambda r:(r-r.mean(0))/r.std(0))(np.apply_along_axis(rankdata,0,v))
    for key in ('mean','z'):
        x=bank[key][index].astype(float);x=x[:,np.isfinite(x).all(0)]
        with np.errstate(all='ignore'):out[key]=np.nanmean([abs(rank(x[y==c]).T@rank(rows.mean_abs_r.to_numpy()[index][y==c]))/(y==c).sum() for c in names],axis=0)
    return out


def dependence_figure(rows,bank,arm,out):
    plt=style();fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained',sharey=True);bins=np.linspace(0,1,26)
    for ax,(title,names) in zip(axes,[('Within CML',CML_PANEL),('Across coupled classes',[n for n in INTER_PANEL if n not in NOISE])]):
        d=strength_dependence(rows,bank,arm,list(names))
        for key,color in (('mean','#D55E00'),('z','#0072B2')):
            v=d[key][np.isfinite(d[key])];ax.hist(v,bins=bins,density=True,histtype='stepfilled',alpha=.45,color=color,
                label=f'{REPRESENTATIONS[key]}: median {np.median(v):.2f}, {np.mean(v>.5):.0%} above .5')
        ax.set(xlabel=r'Within-class $|\rho|$ of a feature with mean $|r|$',title=title,xlim=(0,1));ax.legend(fontsize=7,loc='upper center')
    axes[0].set_ylabel('Density of features')
    return save(fig,out,f'strength-dependence-{arm}')


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({'text.usetex':False,'font.family':'serif','font.serif':['CMU Serif','DejaVu Serif'],'mathtext.fontset':'cm','font.size':9,
        'axes.titlesize':9.5,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
        'legend.frameon':False,'xtick.direction':'out','ytick.direction':'out','figure.dpi':180,'savefig.bbox':'tight','axes.grid':False})
    return plt


def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True);fig.savefig(out/f'{name}.png',dpi=180);fig.savefig(out/f'{name}.svg');return fig


def legend(fig,names,**kw):
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=COLORS[n],markersize=5,label=DISPLAY[n]) for n in names],loc='outside lower center',ncols=min(4,len(names)),**kw)


def strength_figure(rows,out):
    """Design check: observed mean |r| of every recording, by class and arm."""
    plt=style();names=[n for n in CLASSES if n not in NOISE];arms=arms_of(rows)
    fig,axes=plt.subplots(1,len(arms),figsize=(2.3*len(arms),2.9),sharey=True,layout='constrained');rng=np.random.default_rng(0)
    for ax,arm in zip(axes,arms):
        for k,name in enumerate(names):
            v=rows[rows.system.eq(arm)&rows.label.eq(name)].mean_abs_r.to_numpy();ax.scatter(k+rng.uniform(-.28,.28,len(v)),v,s=5,color=COLORS[name],alpha=.6,linewidths=0)
        ax.axhline(rows[rows.label.eq('gaussian-noise')].mean_abs_r.mean(),color='.6',lw=.8,ls=':');ax.set(title=arm,xticks=[],ylim=(0,1))
    axes[0].set_ylabel('Observed mean $|r|$');legend(fig,names)
    return save(fig,out,'strength-by-arm')


def embedding_figure(rows,bank,panel,arms,out,keys=('mean','z'),color='class',cache=None):
    """Rows are arms; columns are PCA and UMAP of each representation. Colour is class, or observed strength."""
    plt=style();columns=[(k,m) for k in keys for m in ('pca','umap')];cache={} if cache is None else cache
    fig,axes=plt.subplots(len(arms),len(columns),figsize=(2.75*len(columns),2.75*len(arms)+.5),layout='constrained',squeeze=False)
    for i,arm in enumerate(arms):
        index=select(rows,arm,panel);part=rows.iloc[index];y=part.label.to_numpy();s=part.mean_abs_r.to_numpy()
        for j,(key,method) in enumerate(columns):
            ax=axes[i,j];e=cache.setdefault((panel,arm,key),embed(bank,index,key));xy=e['scores'][:,:2] if method=='pca' else e['umap']
            if color=='class':
                for name in PANELS[panel]:ax.scatter(*xy[y==name].T,s=9,color=COLORS[name],alpha=.6,linewidths=0)
            else:points=ax.scatter(*xy.T,s=9,c=s,cmap='viridis',vmin=0,vmax=max(.2,np.quantile(s,.98)),alpha=.8,linewidths=0)
            ax.set_box_aspect(1);ax.set(xticks=[],yticks=[])
            for side in ('top','right'):ax.spines[side].set_visible(True)
            if method=='pca':ax.set(xlabel=f'PC1 ({e["evr"][0]:.0%})',ylabel=f'PC2 ({e["evr"][1]:.0%})')
            else:ax.set(xlabel='UMAP 1',ylabel='UMAP 2')
            if i==0:ax.set_title(f'{REPRESENTATIONS[key]}: {method.upper()}')
            if j==0:ax.annotate(arm,(-.32,.5),xycoords='axes fraction',rotation=90,va='center',ha='center',fontsize=10)
            if color!='class' and j==len(columns)-1:fig.colorbar(points,ax=ax,label='Observed mean $|r|$',fraction=.046)
    if color=='class':legend(fig,PANELS[panel])
    return save(fig,out,f'{panel}-embeddings-{color}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract'])
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.stage=='prepare':prepare(a.workers)
    else:extract(a.data,a.output)
