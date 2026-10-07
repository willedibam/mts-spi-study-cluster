"""Dial network: identical noise-driven nodes whose coupling function is the class; strength and timescale are nuisances.

Each node is an AR(1) process, x_i(t) = a x_i(t-1) + k g_i sum_j W_ij phi_beta(x_j(t-tau_ij)/sigma_a) + eps_i(t), eps~N(0,1),
sigma_a=(1-a^2)^-1/2. Inputs run along a random directed acyclic graph (at most two inputs per node), so there are no feedback
loops: no instability or multistability at any gain, and dependence grows monotonically with k. Rows of W have unit norm and
g_i~U[.5,1.5], so pairs differ in strength within a recording. Channels are shuffled and z-scored.
Dials (the classes):
  beta  phi_beta mixes a bounded odd term tanh(u) and a bounded even term exp(-u^2/2), each standardized to zero mean and unit
        variance under N(0,1) and mutually uncorrelated there. beta=0 is near-linear coupling; at beta=1 a driver and its receiver are
        uncorrelated, while two receivers of one driver still correlate.
  lag   tau=1; tau=5; or tau~U{1..9} per edge (same mean lag as 5, variable).
Nuisances, drawn once per recording and independently of class: gain k (log-uniform), node coefficient a~U[.3,.7], the graph.
'uncoupled' is k=0. Sweeps vary one dial at a time for dose-response.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN='dial-network-261007'
DATA=ROOT/'data/proof'/RUN
OUT=ROOT/'results/proof'/RUN
REMOTE='/scratch/ql44/we2614/mts-spi-study/proof/'+RUN
M,T,BURN,DEGREE=16,1000,500,2
SEED,PYSPI_SEED=261121,261122
GAIN,COEFFICIENT=(.25,1.2),(.3,.7)   # gain range from the strength scout: edge dependence rises with gain up to about 1.2, then saturates
LAGS={'lag1':(1,),'lag5':(5,),'lag1-9':tuple(range(1,10))}
CLASSES={f'{shape}-{lag}':dict(beta=beta,lags=LAGS[lag]) for shape,beta in (('odd',0.),('even',1.)) for lag in LAGS}
SWEEPS={**{f'beta{round(100*b):03d}':dict(beta=b,lags=LAGS['lag1']) for b in (0.,.25,.5,.75,1.)},
        **{f'spread{w}':dict(beta=0.,lags=tuple(range(5-w,6+w))) for w in (0,1,2,4)}}
N_CLASS,N_SWEEP=40,24
ODD_SD,EVEN_MEAN,EVEN_SD=.6279,2**-.5,(3**-.5-.5)**.5   # moments of tanh(Z) and exp(-Z^2/2), Z~N(0,1)


def phi(u,beta):
    return ((1-beta)*np.tanh(u)/ODD_SD-beta*(np.exp(-u*u/2)-EVEN_MEAN)/EVEN_SD)/np.hypot(1-beta,beta)


def simulate(beta,lags,gain,a,rng):
    """Returns the (M,T) z-scored recording, the shuffled weight matrix and lags, and oracle strengths."""
    W=np.zeros((M,M));tau=np.zeros((M,M),int)
    for i in range(1,M):
        source=rng.choice(i,min(i,DEGREE),replace=False);w=rng.uniform(.5,1.5,len(source));W[i,source]=w/np.linalg.norm(w);tau[i,source]=rng.choice(lags,len(source))
    W*=rng.uniform(.5,1.5,M)[:,None];sigma=(1-a*a)**-.5;L=max(lags);x=np.zeros((BURN+T+L,M));x[:L]=rng.normal(size=(L,M))*sigma
    edges=[(i,np.flatnonzero(W[i])) for i in range(1,M)];noise=rng.normal(size=x.shape)
    for t in range(L,len(x)):
        x[t]=a*x[t-1]+noise[t]
        for i,s in edges:x[t,i]+=gain*W[i,s]@phi(x[t-tau[i,s],s]/sigma,beta)
    x=x[-T:];x=(x-x.mean(0))/x.std(0)
    oracle=[abs(np.corrcoef(x[tau[i,j]:,i],phi(x[:T-tau[i,j],j],beta))[0,1]) for i,s in edges for j in s] if gain else [0.]
    order=rng.permutation(M);r=np.corrcoef(x.T)[~np.eye(M,dtype=bool)]
    return x[:,order].T,dict(gain=gain,a=a,mean_abs_r=float(abs(r).mean()),edge_dependence=float(np.mean(oracle)))


def tasks():
    out=[('class',name,i) for name in CLASSES for i in range(N_CLASS)]+[('class','uncoupled',i) for i in range(N_CLASS)]
    return out+[('sweep',name,i) for name in SWEEPS for i in range(N_SWEEP)]


def record(task):
    kind,name,instance=task;names=list(CLASSES)+['uncoupled']+list(SWEEPS)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,names.index(name),instance]))
    gain=float(np.exp(rng.uniform(*np.log(GAIN))));a=float(rng.uniform(*COEFFICIENT))
    spec=dict(beta=0.,lags=(1,)) if name=='uncoupled' else {**CLASSES,**SWEEPS}[name]
    x,truth=simulate(spec['beta'],spec['lags'],0. if name=='uncoupled' else gain,a,rng)
    return x,dict(**truth,beta=spec['beta'],lag_mean=float(np.mean(spec['lags'])),lag_spread=int(max(spec['lags'])-min(spec['lags'])))


def scout(workers,n=12):
    """Strength and stability only; no representation is computed."""
    probe=[(k,name,i) for k,name,i in tasks() if i<n]
    with ProcessPoolExecutor(max_workers=workers) as pool:rows=[dict(label=name,**truth,finite=bool(np.isfinite(x).all())) for (k,name,i),(x,truth) in zip(probe,pool.map(record,probe,chunksize=4))]
    f=pd.DataFrame(rows);f['gain_bin']=pd.cut(f.gain,[0,.42,.71,1.21],labels=['low','mid','high'])
    print(f.groupby('label')[['mean_abs_r','edge_dependence']].agg(['min','median','max']).round(3).to_string())
    print(f[f.label.isin(CLASSES)].groupby(['label','gain_bin'],observed=True)[['mean_abs_r','edge_dependence']].median().round(3).unstack().to_string());print('all finite:',f.finite.all())


def prepare(workers):
    import yaml
    from scripts.cross_frequency_locking import partitions
    from scripts.spi_baseline_exploration import sha
    if (DATA/'manifest.json').exists():raise FileExistsError(DATA)
    DATA.mkdir(parents=True,exist_ok=True);rows=[];raw={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (kind,name,instance),(x,truth) in zip(tasks(),pool.map(record,tasks(),chunksize=8)):
            row=f'{kind}-{name}-i{instance:02d}';assert x.shape==(M,T) and np.isfinite(x).all();raw[row]=x
            rows.append(dict(row_id=row,corpus_index=len(rows),M=M,T=T,seed=instance,instance=instance,block=instance,label=name,system=kind,**truth))
    np.savez_compressed(DATA/'observations.npz',**raw,__dataset_names__=np.array(list(raw)),
        __labels_json__=np.array([json.dumps([r['label']]) for r in rows]),__shapes__=np.array([[M,T]]*len(rows)),
        __axis_order__=np.array(['process','observation']))
    manifest=dict(rows=rows,corpus=RUN,archive_sha256=sha(DATA/'observations.npz'),generator_sha256=sha(__file__),
        analysis_scope='Exploratory pilot. Settings fixed by a strength-only scout before p90; see the module docstring.')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config=dict(name=RUN,source=dict(format='named-npz-v1',archive=REMOTE+'/observations.npz',sha256=manifest['archive_sha256'],
        axis_order=['process','observation']),base_output_dir=REMOTE+'/mpis',pyspi_config='configs/pyspi/benchmarked_p90.yaml',
        normalise=False,random_seed=PYSPI_SEED)
    (DATA/'corpus.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    for name,indices in partitions(len(rows)).items():
        (DATA/f'{name}-indices.txt').write_text(''.join(f'{i}\n' for i in indices))
    print(pd.DataFrame(rows).groupby(['system','label'])[['gain','a','mean_abs_r','edge_dependence']].median().round(3).to_string())
    print(len(rows),manifest['archive_sha256'])


# ---- analysis: label-free fits, scored afterwards against class, dials and nuisances ----
PANEL=tuple(CLASSES)+('uncoupled',)
DISPLAY={'odd-lag1':'Near-linear, lag 1','odd-lag5':'Near-linear, lag 5','odd-lag1-9':'Near-linear, lags 1-9','even-lag1':'Even, lag 1','even-lag5':'Even, lag 5',
         'even-lag1-9':'Even, lags 1-9','uncoupled':'Uncoupled'}
COLORS={'odd-lag1':'#56B4E9','odd-lag5':'#0072B2','odd-lag1-9':'#332288','even-lag1':'#E69F00','even-lag5':'#D55E00','even-lag1-9':'#882255','uncoupled':'#777777'}
KEYS={'mean':'Mean of each SPI, $m$','z':'SPI-SPI, $z$','z_ordered':'SPI-SPI, direction kept'}
PAIRS={'beta':('cov_EmpiricalCovariance','mi_kraskov_NN-4'),'spread':('cov_EmpiricalCovariance','xcorr_max_sig-True')}   # named before any outcome was read


def load():
    from scripts import proof_strength_nuisance as P
    rows=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows']);bank=dict(np.load(OUT/'features.npz'))
    np.testing.assert_array_equal(bank['row_id'],rows.row_id);rows['arm']=rows.system;rows['system']=np.where(rows.system.eq('class'),'class',rows.label.str.rstrip('0123456789'))
    return rows,bank,P


def shares(x,y,covariates):
    """Share of total variance between classes, and explained within class by a quadratic in each covariate."""
    total=(x**2).sum();out={'class':0.,**{k:0. for k in covariates}}
    for c in np.unique(y):
        k=y==c;mu=x[k].mean(0);out['class']+=k.sum()*(mu**2).sum()/total
        for name,v in covariates.items():
            d=np.column_stack([np.ones(k.sum()),v[k],v[k]**2]);out[name]+=((d@np.linalg.lstsq(d,x[k]-mu,rcond=None)[0])**2).sum()/total
    return out


def class_table(rows,bank,P,names=PANEL):
    """One embedding per representation of the class recordings, scored against class, each dial, and the nuisances."""
    from scipy.stats import spearmanr
    index=np.flatnonzero(rows.arm.eq('class').to_numpy()&rows.label.isin(names).to_numpy());part=rows.iloc[index];y=part.label.to_numpy();inst=part.instance.to_numpy()
    shape=np.array([v.split('-')[0] for v in y]);lag=np.array([v.split('-',1)[-1] for v in y]);coupled=y!='uncoupled';gain=part.gain.to_numpy();out=[]
    strata=np.where(~coupled,'none',np.where(gain<np.quantile(gain[coupled],1/3),'low',np.where(gain<np.quantile(gain[coupled],2/3),'mid','high')))
    for key,scores in [('mean |r|',part.mean_abs_r.to_numpy()[:,None])]+[(k,P.embed(bank,index,k,umap=False)['scores']) for k in KEYS]:
        g=P.geometry(scores,y,part.mean_abs_r.to_numpy(),coupled,inst);x=scores-scores.mean(0)
        from sklearn.neighbors import NearestNeighbors
        near=NearestNeighbors(n_neighbors=6).fit(scores).kneighbors(scores,return_distance=False)[:,1:];agree=lambda v,m:float(np.mean((v[near]==v[:,None])[m]))
        sh=shares(x[coupled],y[coupled],dict(gain=np.log(gain[coupled]),a=part.a.to_numpy()[coupled]))
        out.append(dict(representation=key,silhouette=g['silhouette'],class_agreement=g['purity'],shape_agreement=agree(shape,coupled),lag_agreement=agree(lag,coupled),
            **{f'class_agreement_{t}_gain':agree(y,strata==t) for t in ('low','mid','high')},ceiling=g['ceiling'],
            share_class=sh['class'],share_gain=sh['gain'],share_a=sh['a'],rho_pc1_gain=abs(spearmanr(scores[coupled,0],gain[coupled]).statistic)))
    return pd.DataFrame(out).set_index('representation').round(2)


def coordinate(bank,first,second):
    order=[str(v) for v in bank['spi_order']];i,j=sorted((order.index(first),order.index(second)));n=len(order)
    return bank['z'][:,i*n-i*(i+1)//2+j-i-1],bank['mean'][:,order.index(first)],bank['mean'][:,order.index(second)]


def sweep_table(rows,bank,P):
    """Does the leading label-free coordinate, or one named SPI pair, follow the dial rather than the gain?"""
    from scipy.stats import spearmanr
    rho=lambda u,v:round(float(abs(spearmanr(u,v,nan_policy='omit').statistic)),2);out=[]
    for sweep,dial in (('beta','beta'),('spread','lag_spread')):
        index=np.flatnonzero(rows.system.eq(sweep).to_numpy());part=rows.iloc[index];d,gain,a=(part[c].to_numpy() for c in (dial,'gain','a'))
        z,first,second=(v[index] for v in coordinate(bank,*PAIRS[sweep]))
        readouts={'mean |r|':part.mean_abs_r.to_numpy(),**{f'{k} PC{c+1}':P.embed(bank,index,k,umap=False)['scores'][:,c] for k in ('mean','z') for c in (0,1)},
                  f'z({PAIRS[sweep][0]}, {PAIRS[sweep][1]})':z,f'mean {PAIRS[sweep][0]}':first,f'mean {PAIRS[sweep][1]}':second}
        out+= [dict(sweep=sweep,readout=name,dial=rho(v,d),gain=rho(v,gain),a=rho(v,a)) for name,v in readouts.items()]
    return pd.DataFrame(out)


def design_figure(rows,P):
    """Observed mean |r| against gain for the class recordings: strength overlaps across classes."""
    plt=P.style();fig,ax=plt.subplots(figsize=(4.6,3.2),layout='constrained');part=rows[rows.arm.eq('class')]
    for name in PANEL:
        k=part.label.eq(name);ax.scatter(part.gain[k] if name!='uncoupled' else np.full(k.sum(),GAIN[0]*.85),part.mean_abs_r[k],s=9,color=COLORS[name],alpha=.7,linewidths=0,label=DISPLAY[name])
    ax.set(xscale='log',xlabel='Gain $k$ (uncoupled shown at left)',ylabel='Observed mean $|r|$');ax.legend(fontsize=6.5,ncols=2,loc='upper left')
    return P.save(fig,OUT,'strength-by-gain')


def sweep_figure(rows,bank,P):
    plt=P.style();fig,axes=plt.subplots(2,2,figsize=(7.4,5.6),layout='constrained');rng=np.random.default_rng(0)
    for column,(sweep,dial,label) in enumerate((('beta','beta',r'Even share of the coupling, $\beta$'),('spread','lag_spread','Spread of lags around 5'))):
        index=np.flatnonzero(rows.system.eq(sweep).to_numpy());part=rows.iloc[index];d=part[dial].to_numpy();z,_,second=(v[index] for v in coordinate(bank,*PAIRS[sweep]))
        jitter=d+rng.uniform(-.12,.12,len(d))*np.diff(np.unique(d)).min()
        for ax,(v,name) in zip(axes[:,column],((z,f'$z$({PAIRS[sweep][0]},\n{PAIRS[sweep][1]})'),(second,f'Mean of {PAIRS[sweep][1]}'))):
            points=ax.scatter(jitter,v,c=part.gain,cmap='viridis',s=10,alpha=.85,linewidths=0);ax.set(xlabel=label,ylabel=name,xticks=np.unique(d))
    fig.colorbar(points,ax=axes,label='Gain $k$',fraction=.03)
    return P.save(fig,OUT,'dose-response')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['scout','prepare','extract'])
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.stage=='scout':scout(a.workers)
    elif a.stage=='prepare':prepare(a.workers)
    else:
        from scripts.proof_strength_nuisance import extract
        extract(DATA,OUT,run=RUN,ordered=True)
