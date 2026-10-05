"""Physics-only finite-population check of Abrams et al. PRL101 084103 Eq1/12.

No SPI superiority is presumed. Std(r2) is a breathing-amplitude diagnostic,
not a newly claimed thermodynamic order parameter. Total M=N counts oscillators.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scripts.spi_baseline_exploration import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/order-parameter-inference/dependence-transition-261005/chimera-physics'


def reduced(state,A,beta=.1):
    r,psi=state;mu=(1+A)/2;nu=(1-A)/2;alpha=np.pi/2-beta
    return np.array([(1-r*r)/2*(mu*r*np.cos(alpha)+nu*np.cos(psi-alpha)),
        (1+r*r)/(2*r)*(mu*r*np.sin(alpha)-nu*np.sin(psi-alpha))
        -mu*np.sin(alpha)-nu*r*np.sin(psi+alpha)])


def hopf(beta=.1):
    psi=-.5*np.arcsin(2*np.sin(2*beta))
    r=np.sqrt(np.sin(2*beta+psi)/(np.sin(2*beta-psi)+2*np.sin(psi)))
    A=(np.sin(beta+psi)+r*np.sin(beta))/(np.sin(beta+psi)-r*np.sin(beta))
    return float(A),float(r),float(psi)


@njit(cache=True)
def velocity(theta,A,beta=.1):
    n=len(theta)//2;mu=(1+A)/2;nu=(1-A)/2;alpha=np.pi/2-beta
    z1=np.exp(1j*theta[:n]).mean();z2=np.exp(1j*theta[n:]).mean()
    out=np.empty(len(theta))
    for i in range(len(theta)):
        h=mu*z1+nu*z2 if i<n else nu*z1+mu*z2
        out[i]=1.+(h*np.exp(-1j*(theta[i]+alpha))).imag
    return out


@njit(cache=True)
def integrate(theta,A,dt,burn_steps,sample_steps,record_count):
    n=len(theta)//2;raw=np.empty((len(theta),record_count));r=np.empty((2,record_count))
    t=theta.copy();index=0
    for k in range(burn_steps+sample_steps*record_count):
        a=velocity(t,A);b=velocity(t+.5*dt*a,A)
        c=velocity(t+.5*dt*b,A);d=velocity(t+dt*c,A)
        t+=dt/6*(a+2*b+2*c+d)
        t=(t+np.pi)%(2*np.pi)-np.pi
        if k>=burn_steps and (k-burn_steps+1)%sample_steps==0:
            raw[:,index]=np.sin(t)
            r[0,index]=abs(np.exp(1j*t[:n]).mean())
            r[1,index]=abs(np.exp(1j*t[n:]).mean());index+=1
    return raw,r


def initial(M,seed,kind):
    rng=np.random.default_rng(np.random.SeedSequence([261061,seed]));n=M//2
    fixed=root(lambda s:reduced(s,.2),[.73,-.25])
    if not fixed.success or np.linalg.norm(fixed.fun)>1e-8:raise ValueError('Initial OA branch solve failed')
    r,psi=fixed.x
    u=(np.arange(n)+rng.random(n))/n*2*np.pi if kind=='stratified' else rng.uniform(0,2*np.pi,n)
    v=np.exp(1j*u);second=np.angle((r+v)/(1+r*v))-psi
    return np.r_[rng.normal(0,.001,n),second]


def case(task):
    M,A,seed,kind,dt,burn,reference=task
    raw,r=integrate(initial(M,seed,kind),A,dt,round(burn/dt),round(1/dt),1000+reference)
    x=raw[:,:1000];future=r[:,1000:];halves=np.array_split(future[1],2)
    sd=x.std(1);valid=sd>1e-12
    if not valid.all():mean_r=np.nan
    else:
        normalized=(x-x.mean(1,keepdims=True))/sd[:,None]
        mean_r=((normalized.sum(0)**2).mean()-M)/(M*(M-1))
    row=dict(M=M,N=M,T=1000,A=A,beta=.1,seed=seed,initialization=kind,dt=dt,burn=burn,
        reference=reference,Q_breathing=float(future[1].std()),Q_first=float(halves[0].std()),
        Q_second=float(halves[1].std()),mean_r2=float(future[1].mean()),min_r2=float(future[1].min()),
        mean_r1=float(future[0].mean()),mean_Pearson=float(mean_r),min_channel_sd=float(sd.min()))
    # Retain full M=N observations only for affordable prospective SPI candidates.
    return row,x if M<=32 else np.empty((0,0)),r


def run(out,workers,refine=False,smoke=False):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'rows.csv').exists():raise FileExistsError(out/'rows.csv')
    sizes=[16,32,64,256];controls=np.linspace(.23,.31,9)
    tasks=[(m,float(a),s,k,.025 if refine else .05,4000 if refine else 2000,4000 if refine else 2000)
        for m in sizes for a in controls for s in range(4) for k in ['stratified','random']]
    if smoke:tasks=[t for t in tasks if t[0] in [16,256] and t[1] in [controls[0],controls[-1]] and t[2]==0]
    oa=[]
    init=root(lambda s:reduced(s,.2),[.73,-.25]).x
    for A in controls:
        solution=solve_ivp(lambda t,s:reduced(s,A),(0,10000),init,t_eval=np.arange(8001,10001),
                           method='DOP853',rtol=1e-9,atol=1e-11)
        if not solution.success:raise RuntimeError(solution.message)
        oa.append(dict(A=A,Q_breathing=solution.y[0].std(),mean_r2=solution.y[0].mean()))
    pd.DataFrame(oa).to_csv(out/'continuum.csv',index=False)
    rows=[];arrays={};traces={}
    integrate(np.zeros(16),.28,.05,0,1,1)  # Compile once before forking Linux workers.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for j,(row,x,r) in enumerate(pool.map(case,tasks)):
            key=f"M{row['M']}-A{row['A']:.3f}-s{row['seed']}-{row['initialization']}"
            row['row_id']=key;rows.append(row)
            if x.size:arrays[key]=x
            traces[key]=r
            if j%16==0:print('physics',j+1,'/',len(tasks),flush=True)
    pd.DataFrame(rows).to_csv(out/'rows.csv',index=False)
    np.savez_compressed(out/'observations.npz',**arrays)
    np.savez_compressed(out/'coherence-traces.npz',**traces)
    (out/'report.json').write_text(json.dumps(dict(records=len(rows),hopf=hopf(),source_sha256=sha(__file__),
        scope='Physics only. Continuum OA boundary is not asserted exact at finite N. Initial-condition families kept separate. No p90, no claimed baseline advantage.',
        observation='sin(theta), one process per oscillator; omega=1; first1000 samples observed, subsequent reference is disjoint.'),indent=2)+'\n')
    print(pd.DataFrame(rows).groupby(['M','initialization','A'])[['Q_breathing','mean_r2','mean_Pearson']].mean().to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--refine',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.output,a.workers,a.refine,a.smoke)
