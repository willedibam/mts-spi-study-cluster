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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['scout','prepare','extract'])
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.stage=='scout':scout(a.workers)
    elif a.stage=='prepare':prepare(a.workers)
    else:
        from scripts.proof_strength_nuisance import extract
        extract(DATA,OUT,run=RUN,ordered=True)
