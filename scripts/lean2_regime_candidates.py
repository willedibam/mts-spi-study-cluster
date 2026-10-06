"""Physics-first scouts for CGLE defects and noise-driven rate-network chaos.

Independent of the concurrently developed cross-frequency experiment. All generator
parameters and numerical resolutions are recorded per case. No SPI outcomes select
the physical sweep. CGLE uses exact local flow + spectral Strang splitting; the rate
network uses additive-noise Heun with its matching tangent map.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
RUN = 'lean2-regime-candidates-261006'
REMOTE = '/scratch/ql44/we2614/mts-spi-study/order-parameter-inference/' + RUN


def local_flow(a, h, c3):
    denominator = 1 + abs(a)**2 * np.expm1(2*h)
    return a * np.exp(h + (-.5 + .5j*c3)*np.log(denominator))


def cgle_step(a, dt, c1, c3, k):
    a = local_flow(a, dt/2, c3)
    a = np.fft.ifft(np.fft.fft(a)*np.exp(-(1+1j*c1)*k*k*dt))
    return local_flow(a, dt/2, c3)


def defect_charges(old, new):
    """Oriented space-time plaquette winding, including cancelling defect pairs."""
    horizontal_old = np.angle(np.roll(old, -1)*old.conj())
    horizontal_new = np.angle(np.roll(new, -1)*new.conj())
    vertical = np.angle(new*old.conj())
    return np.rint((horizontal_old + np.roll(vertical, -1) - horizontal_new - vertical)/(2*np.pi)).astype(int)


def cgle(control, seed, L=128., dx=.5, dt=.025, burn=2000., reference=4000., T=1000, sample=.5, M=16):
    n = round(L/dx)
    if n < M or abs(n*dx-L)>1e-10: raise ValueError('invalid spatial grid')
    rng = np.random.default_rng(np.random.SeedSequence([261106, seed, n, round(control*1e6)]))
    a = 1 + rng.uniform(-.1,.1,n) + 1j*rng.uniform(-.1,.1,n)
    k = 2*np.pi*np.fft.fftfreq(n, d=dx)
    propagator = np.exp(-(1+3.5j)*k*k*dt)
    sites = np.linspace(0,n,M,endpoint=False,dtype=int)
    every = round(sample/dt)
    assert abs(every*dt-sample)<1e-10
    nb, no, nr = round(burn/dt), T*every, round(reference/dt)
    observed = np.empty((M,T)); counts = np.zeros(2, dtype=int)
    obs_defects = 0; minimum = 1e9; snapshots=[]; net=0
    for step in range(nb+no+nr):
        old = a
        a = local_flow(a,dt/2,control)
        a = np.fft.ifft(np.fft.fft(a)*propagator)
        a = local_flow(a,dt/2,control)
        if step >= nb:
            charges = defect_charges(old,a); count = abs(charges).sum()
            if step < nb+no:
                obs_defects += count
                if (step-nb+1)%every==0: observed[:,(step-nb+1)//every-1]=a.real[sites]
                if step-nb in (0,no//2,no-1): snapshots.append(a.copy())
            else:
                counts[min(1,2*(step-nb-no)//nr)] += count
                net += charges.sum(); minimum = min(minimum,float(abs(a).min()))
    durations = np.array([(nr+1)//2,nr//2])*dt
    truth = dict(Q=float(counts.sum()/(L*nr*dt)),Q_first=float(counts[0]/(L*durations[0])),
        Q_second=float(counts[1]/(L*durations[1])),defects=int(counts.sum()),net_defects=int(net),
        Q_observed=float(obs_defects/(L*T*sample)),min_amplitude=minimum,N=n,L=L,dx=dx,dt=dt)
    return observed,truth,dict(snapshots=np.array(snapshots),sites=sites)


@njit(cache=True)
def rate_integrate(w, g, sigma, seed, dt, nb, no, nr, every, M):
    np.random.seed(seed)
    n=len(w); x=np.random.normal(0,.1,n); v=np.random.normal(0,1,n); v/=np.linalg.norm(v)
    observed=np.empty((M,no//every)); logs=np.zeros(2); state_variance=0.; renorm=0
    # Jii=0; row i receives from column j. Input is independent per neuron.
    j=g*w
    for step in range(nb+no+nr):
        tx=np.tanh(x); f=-x+j@tx
        vf=-v+j@((1-tx*tx)*v)
        noise=np.sqrt(2*dt)*sigma*np.random.normal(0,1,n)
        xp=x+dt*f+noise; vp=v+dt*vf
        tp=np.tanh(xp)
        x=x+.5*dt*(f-xp+j@tp)+noise
        v=v+.5*dt*(vf-vp+j@((1-tp*tp)*vp))
        norm=np.linalg.norm(v)
        if step>=nb+no:
            h=min(1,2*(step-nb-no)//nr); logs[h]+=np.log(norm)
            state_variance+=np.mean(x*x)
        v/=norm; renorm+=1
        if step>=nb and step<nb+no and (step-nb+1)%every==0:
            for s in range(M): observed[s,(step-nb+1)//every-1]=x[s*n//M]
    return observed,logs,state_variance/nr


def rate(control, seed, N=16, dt=.02, burn=1000., reference=3000., T=1000, sample=.5, M=16, sigma=np.sqrt(.125)):
    rng=np.random.default_rng(np.random.SeedSequence([261107,seed,N]))
    w=rng.normal(size=(N,N))/np.sqrt(N);np.fill_diagonal(w,0)
    every=round(sample/dt);assert abs(every*dt-sample)<1e-10 and N>=M
    nb,no,nr=round(burn/dt),T*every,round(reference/dt)
    noise_seed=int(np.random.SeedSequence([261108,seed,N,round(control*1e6)]).generate_state(1)[0])
    x,logs,variance=rate_integrate(w,control,sigma,noise_seed,dt,nb,no,nr,every,M)
    durations=np.array([(nr+1)//2,nr//2])*dt
    return x,dict(Q=float(logs.sum()/(nr*dt)),Q_first=float(logs[0]/durations[0]),Q_second=float(logs[1]/durations[1]),
        state_variance=float(variance),N=N,dt=dt,sigma=sigma),dict(connectivity_sha256=np.array(hashlib.sha256(w.tobytes()).hexdigest()))


def plan(path, kind='initial'):
    tasks=[]
    for L in (128.,512.):
        for c in (.6,.7,.75,.8,.9,1.):
            for seed in range(4):
                tasks.append(dict(system='cgle',control=c,seed=seed,L=L))
    for N in (16,64,256):
        for g in (1.1,1.3,1.45,1.6,1.8,2.):
            for seed in range(4): tasks.append(dict(system='rate',control=g,seed=seed,N=N))
    if kind=='refinement':
        tasks=[]
        for c in (.8,.85,.9):
            for dx,dt in ((.5,.025),(.5,.0125),(.25,.0125)):
                for seed in range(4):tasks.append(dict(system='cgle',control=c,seed=seed,L=512.,dx=dx,dt=dt))
        for g in (1.6,1.8):
            for seed in range(4):tasks.append(dict(system='rate',control=g,seed=seed,N=256,dt=.01))
        for g in (1.4,1.6,1.8):
            for seed in range(4):tasks.append(dict(system='rate',control=g,seed=seed,N=1024))
    if kind=='production-cgle':
        tasks=[dict(system='cgle',control=float(round(c,4)),seed=seed,L=512.,dx=.5,dt=.0125,
                    burn=4000.,reference=8000.,M=16,T=1000,sample=.5)
               for c in np.linspace(.74,.90,17) for seed in range(100,116)]
    if kind=='production-rate':
        tasks=[dict(system='rate',control=float(round(g,4)),seed=seed,N=256,dt=.01,
                    burn=1000.,reference=4000.,M=16,T=1000,sample=.5)
               for g in np.linspace(1.5,1.9,17) for seed in range(100,116)]
    if kind=='production-rate1024':
        tasks=[dict(system='rate',control=float(round(g,4)),seed=seed,N=1024,dt=.01,
                    burn=1000.,reference=4000.,M=16,T=1000,sample=.5)
               for g in np.linspace(1.35,1.75,17) for seed in range(100,116)]
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(dict(tasks=tasks,purpose='Physics scout only; no SPI-based selection',kind=kind,
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    path.with_suffix('.indices.txt').write_text(''.join(f'{i}\n' for i in range(len(tasks))))


def run_case(plan_path,index,out):
    task=json.loads(plan_path.read_text())['tasks'][index];params=task.copy();system=params.pop('system')
    target=out/f'case-{index:04d}.npz';meta=out/f'case-{index:04d}.json'
    if target.exists() or meta.exists():raise FileExistsError(target)
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    x,truth,extra=(cgle if system=='cgle' else rate)(**params)
    if not np.isfinite(x).all() or np.any(x.std(1)<1e-8): raise ValueError('Invalid/constant observations')
    r=np.corrcoef(x);off=~np.eye(len(x),dtype=bool)
    metadata={**task,**truth,'M':len(x),'T':x.shape[1],
        'mean_abs_r':float(abs(r[off]).mean()),'mean_r':float(r[off].mean()),'rank':int(np.linalg.matrix_rank(x)),
        'seconds':time.monotonic()-start,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    np.savez_compressed(target,observations=x,**extra)
    meta.write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata),flush=True)


def audit_cases(plan_path,indices_path,out):
    tasks=json.loads(plan_path.read_text())['tasks']
    indices=[int(x) for x in indices_path.read_text().split()]
    assert len(indices)==len(set(indices))
    for index in indices:
        row=json.loads((out/f'case-{index:04d}.json').read_text())
        for key,value in tasks[index].items():assert row[key]==value,(index,key,row[key],value)
        x=np.load(out/f'case-{index:04d}.npz')['observations']
        assert x.shape==(row['M'],row['T']) and np.isfinite(x).all()
        assert np.isfinite([row['Q'],row['Q_first'],row['Q_second']]).all()
    print(f'Audited {len(indices)} complete physics cases',flush=True)


def summarize(out):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=pd.DataFrame([json.loads(p.read_text()) for p in sorted(out.glob('case-*.json'))])
    rows.to_csv(out/'physics.csv',index=False)
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'legend.frameon':False,'lines.linewidth':1.7,'lines.markersize':2.7})
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    for i,system in enumerate(('cgle','rate')):
        data=rows[rows.system.eq(system)]
        for n,frame in data.groupby('N'):
            group=frame.groupby('control')
            for j,col in enumerate(('Q','mean_abs_r')):
                avg=group[col].mean();axes[i,j].plot(avg.index,avg,'o-',label=f'$N={int(n)}$')
                axes[i,j].fill_between(avg.index,group[col].min(),group[col].max(),alpha=.12,lw=0)
        axes[i,0].set_ylabel('Defects / (length × time)' if system=='cgle' else 'Conditional Lyapunov exponent')
        axes[i,1].set_ylabel('Mean absolute Pearson')
        for ax in axes[i]: ax.set_xlabel('$c_3$' if system=='cgle' else '$g$');ax.legend()
        if system=='rate':axes[i,0].axhline(0,color='.5',lw=.7,ls=':')
    fig.suptitle('Physics scouts · M=16, T=1000 · means; bands span 4 realizations')
    for ext in ('png','svg'):fig.savefig(out/f'physics-scout.{ext}',dpi=180)
    print(rows.groupby(['system','N','control'])[['Q','Q_first','Q_second','mean_abs_r','seconds']].mean().round(5).to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','case','summarize','audit'])
    p.add_argument('--plan',type=Path);p.add_argument('--index',type=int);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--indices',type=Path)
    p.add_argument('--plan-kind',choices=['initial','refinement','production-cgle','production-rate','production-rate1024'],default='initial')
    a=p.parse_args()
    if a.stage=='plan':plan(a.out,a.plan_kind)
    elif a.stage=='case':run_case(a.plan,a.index,a.out)
    elif a.stage=='audit':audit_cases(a.plan,a.indices,a.out)
    else:summarize(a.out)
