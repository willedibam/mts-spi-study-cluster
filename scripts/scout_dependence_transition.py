"""Literature-defined GS physics and cheap dependency probes; no pyspi here."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numba import njit
from scipy.signal import hilbert
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/analysis/dependence-transition-261005.yaml'
OUT = ROOT/'results/order-parameter-inference/dependence-transition-261005'


@njit
def rhs(s, k, wd, heterogeneous):
    d = np.empty(12)
    if heterogeneous:
        d[0] = -6*(s[1]+s[2])
        d[1] = 6*(s[0]+.2*s[1])
        d[2] = 6*(.2+s[2]*(s[0]-5.7))
        for j in (3, 6):
            x,y,z=s[j:j+3]
            d[j]=10*(y-x)
            d[j+1]=28*x-y-x*z+k*s[1]
            d[j+2]=x*y-(8/3)*z
        x,y,z=s[3:6];u,v,w=s[9:12]
        d[9]=10*(v-u)
        d[10]=(28-z)*u-v-x*w
        d[11]=y*u+x*v-(8/3)*w
    else:
        d[0]=-wd*s[1]-s[2]
        d[1]=wd*s[0]+.15*s[1]
        d[2]=.2+s[2]*(s[0]-10)
        for j in (3, 6):
            x,y,z=s[j:j+3]
            d[j]=-.95*y-z+k*(s[0]-x)
            d[j+1]=.95*x+.15*y
            d[j+2]=.2+z*(x-10)
        x,y,z=s[3:6];u,v,w=s[9:12]
        d[9]=-k*u-.95*v-w
        d[10]=.95*u+.15*v
        d[11]=z*u+(x-10)*w
    return d


@njit
def integrate(initial, k, wd, heterogeneous, dt, stride, burn, reference, t=1000):
    s=initial.copy(); s[9:12]/=np.linalg.norm(s[9:12])
    x=np.empty((6,t)); cle=np.zeros(2); err=np.zeros(2)
    mean=np.zeros((2,3)); square=np.zeros(2); counts=np.zeros(2)
    start=burn+t*stride
    for i in range(start+reference):
        a=rhs(s,k,wd,heterogeneous)
        b=rhs(s+dt*a/2,k,wd,heterogeneous)
        c=rhs(s+dt*b/2,k,wd,heterogeneous)
        d=rhs(s+dt*c,k,wd,heterogeneous)
        s+=dt*(a+2*b+2*c+d)/6
        norm=np.linalg.norm(s[9:12]);s[9:12]/=norm
        if burn<=i<start and (i-burn+1)%stride==0:
            x[:,(i-burn+1)//stride-1]=s[:6]
        if i>=start:
            h=min(1,2*(i-start)//reference)
            cle[h]+=np.log(norm);counts[h]+=1
            err[h]+=np.sum((s[3:6]-s[6:9])**2)
            mean[h]+=s[3:6];square[h]+=np.sum(s[3:6]**2)
    cle/=counts*dt
    variance=square/counts-np.sum((mean/counts[:,None])**2,axis=1)
    aux=np.sqrt(err/counts/variance)
    return x,cle,aux


@njit
def maps(initial,k,burn,reference,t=1000):
    # Auxiliary copies are diagnostics, never observed channels.
    x,y,a=initial[0].copy(),initial[1].copy(),initial[2].copy()
    modules=len(x);observed=np.empty((2*modules,t));cle=np.zeros(2)
    error=np.zeros(2);ydiff=np.zeros(2);ym=np.zeros((2,modules));ys=np.zeros(2)
    counts=np.zeros(2)
    for i in range(burn+t+reference):
        if i>=burn+t:
            h=min(1,2*(i-burn-t)//reference)
            cle[h]+=np.sum(np.log(np.maximum(np.abs((1-k)*4*(1-2*y)),1e-300)))
            error[h]+=np.sum((y-a)**2);ydiff[h]+=np.sum((x-y)**2)
            ym[h]+=y;ys[h]+=np.sum(y*y);counts[h]+=1
        fx=4*x*(1-x)
        y=(1-k)*4*y*(1-y)+k*fx
        a=(1-k)*4*a*(1-a)+k*fx
        x=fx
        if burn<=i<burn+t:
            observed[::2,i-burn]=x;observed[1::2,i-burn]=y
    variance=ys/counts-np.sum((ym/counts[:,None])**2,axis=1)
    return observed,cle/(counts*modules),np.sqrt(error/counts/variance),np.sqrt(ydiff/counts/variance)


def normalize(x):
    sd=x.std(axis=1,keepdims=True)
    if not np.isfinite(x).all() or (sd<1e-10).any():
        raise ValueError('Nonfinite or constant observed state')
    return (x-x.mean(axis=1,keepdims=True))/sd


def probes(x):
    x=normalize(x);m,t=x.shape;rank=normalize(rankdata(x,axis=1))
    cov=x@x.T/t
    values={'Pearson':cov,'absolute_Pearson':abs(cov),'Pearson_squared':cov**2,
            'Spearman':rank@rank.T/t}
    for lag in (1,5):
        values[f'Pearson_lag{lag}']=normalize(x[:,lag:])@normalize(x[:,:-lag]).T/(t-lag)
        values[f'Spearman_lag{lag}']=normalize(rank[:,lag:])@normalize(rank[:,:-lag]).T/(t-lag)
    square=normalize(x*x)
    values['square_to_linear']=square@x.T/t
    values['square_to_square']=square@square.T/t
    phase=np.exp(1j*np.angle(hilbert(x,axis=1)))
    values['PLV']=abs(phase@phase.conj().T/t)
    edge=np.array([v[~np.eye(m,dtype=bool)] for v in values.values()])
    z=np.corrcoef(edge)[np.triu_indices(len(edge),1)]
    return np.mean(edge,axis=1),z,list(values)


def case(args):
    name,cfg,k,seed,base,refine=args
    rng=np.random.default_rng(np.random.SeedSequence([base,seed]))
    if name=='logistic-modules':
        initial=rng.uniform(.05,.95,(3,cfg['M']//2))
        raw,cle,aux,identical=maps(initial,k,cfg['burn_steps'],cfg['reference_steps']*refine)
    else:
        hetero=name=='rossler-lorenz'
        initial=rng.uniform(-1,1,12)
        initial[[2,5,8]]=[.2,20 if hetero else .2,22 if hetero else .3]
        dt=cfg['dt']/refine
        raw,cle,aux=integrate(initial,k,cfg.get('omega_drive',1.3),hetero,dt,
            round(cfg['sample_dt']/dt),round(cfg['burn_time']/dt),round(cfg['reference_time']/dt))
        identical=np.full(2,np.nan)
    mean,z,names=probes(raw)
    row=dict(system=name,control=float(k),seed=seed,M=cfg['M'],T=1000,refinement=refine,
             CLE=float(cle.mean()),CLE_first=float(cle[0]),CLE_second=float(cle[1]),
             aux_error=float(aux.mean()),aux_first=float(aux[0]),aux_second=float(aux[1]),
             identical_error=float(identical.mean()),mean_r=float(mean[0]),mean_abs_r=float(mean[1]))
    return row,mean,z,raw,names


def run(output,workers,refine=1):
    config=yaml.safe_load(CONFIG.read_text())
    if (output/'rows.csv').exists():raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    tasks=[(name,c,k,seed,config['seed'],refine) for name,c in config['candidates'].items()
           for k in c['controls'] for seed in range(config['physics_seeds'])]
    rows=[];means=[];zs=[];waveforms={};names=None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(r,m,z,x,names) in enumerate(pool.map(case,tasks)):
            key=f"{r['system']}-k{r['control']:.6f}-s{r['seed']:02d}"
            rows.append(dict(row_id=key,**r));means.append(m);zs.append(z);waveforms[key]=normalize(x)
            if i%32==0:print('physics',i+1,'/',len(tasks),flush=True)
    pd.DataFrame(rows).to_csv(output/'rows.csv',index=False)
    np.savez_compressed(output/'probes.npz',mean=means,z=zs,names=np.array(names))
    np.savez_compressed(output/'observations.npz',**waveforms)
    report=dict(records=len(rows),config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                refinement=refine,scope='Exploratory physics and 11 cheap probes, NOT p90; no outcome-selected components.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(pd.DataFrame(rows).groupby(['system','control'])[['CLE','aux_error','mean_r','mean_abs_r']].mean().round(5).to_string())


def check_numerics(output,workers):
    config=yaml.safe_load(CONFIG.read_text());tasks=[]
    selections={'rossler-detuned':[.09,.11,.14], 'rossler-close':[.09,.12,.14],
                'rossler-lorenz':[6.,6.5,6.75,7.], 'logistic-modules':[.30,.33,.35,.52]}
    for name,controls in selections.items():
        cfg=config['candidates'][name].copy()
        if name=='logistic-modules':cfg['burn_steps']*=2;cfg['reference_steps']*=2
        else:cfg['burn_time']*=5;cfg['reference_time']*=5
        for k in controls:
            for seed in range(4):tasks.append((name,cfg,k,seed,config['seed'],2))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows=[r[0] for r in pool.map(case,tasks)]
    pd.DataFrame(rows).to_csv(output/'numerical-refinement.csv',index=False)
    print(pd.DataFrame(rows).groupby(['system','control'])[['CLE','aux_error']].agg(['mean','std']).round(6).to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT/'physics')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--refine',type=int,default=1)
    p.add_argument('--check-numerics',action='store_true')
    a=p.parse_args()
    if a.check_numerics:check_numerics(a.output,a.workers)
    else:run(a.output,a.workers,a.refine)
