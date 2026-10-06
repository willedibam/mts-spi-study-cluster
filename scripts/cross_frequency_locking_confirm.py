"""M=48 confirmation (and M=24 repeat) of the 2:1 locking sweep under recording-specific nuisances; protocol fixed before p90.

Same dynamics and truth as scripts/cross_frequency_locking.py with 16 oscillators per community and fresh seeds.
Every arm observes each channel through white sensor noise. One nuisance per arm is drawn once per recording,
independently of the control:
  random-noise       sensor-noise SD eta~U[.3,.9]; confirms the M=24 result.
  common-mode        eta=.5 plus a shared white signal of SD U[0,.8] on every channel (reference or common pickup);
                     it inflates every dependence estimate where sensor noise attenuates it.
  internal-coupling  eta=.5 and within-community coupling eps~U[.15,.6]; declared boundary test. This changes the
                     dependence structure itself, not its level or scale, so z-PC1 is expected to be contaminated.
Declared before p90: centered z-PC1 is primary; held |Spearman| with Q and with the nuisance are the headline scores.
"""
import argparse
from functools import partial
from pathlib import Path
import numpy as np
from scripts.cross_frequency_locking import ROOT,EPS,simulate
from scripts import cross_frequency_locking_snr as snr

ARMS=('random-noise','common-mode','internal-coupling')
# m48 is the declared confirmation. m24 repeats the same arms on separate fresh seeds at the original size; it was
# added when the first m48 submission was killed for memory, before any outcome of either run had been read.
RUNS={'m48':dict(run='cross-frequency-locking-confirm-261006',n=16,first=dict(zip(ARMS,(300,400,500))),noise_seed=261094,pyspi_seed=261095),
      'm24':dict(run='cross-frequency-locking-nuisance-261006',n=8,first=dict(zip(ARMS,(1000,1100,1200))),noise_seed=261096,pyspi_seed=261097)}


def record(task,n=16,noise_seed=261094):
    arm,gamma,seed=task;eta,common,eps=.5,0.,EPS
    rng=np.random.default_rng(np.random.SeedSequence([noise_seed,ARMS.index(arm),int(round(gamma*1000)),seed]))
    level=float(rng.uniform())
    if arm==ARMS[0]:eta=.3+.6*level
    if arm==ARMS[1]:common=.8*level
    if arm==ARMS[2]:eps=.15+.45*level
    raw,truth=simulate(gamma,seed,n=n,eps=eps)
    x,truth=snr.observe(raw,truth,rng,eta,common)
    truth.update(common=common,eps=eps,nuisance=dict(zip(ARMS,(eta,common,eps)))[arm])
    return x,truth


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract','analyze','figure','scatter'])
    p.add_argument('--size',choices=list(RUNS),default='m48');p.add_argument('--data',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args();c=RUNS[a.size];RUN=c['run']
    a.data=a.data or ROOT/'data/order-parameter-inference'/RUN;a.output=a.output or ROOT/'results/order-parameter-inference'/RUN
    names=dict(stem='confirm-comparison',heading='2:1 locking under recording-specific nuisances')
    if a.stage=='prepare':snr.prepare(a.workers,RUN,ARMS,c['first'],partial(record,n=c['n'],noise_seed=c['noise_seed']),c['n'],c['pyspi_seed'],__file__)
    else:
        a.output.mkdir(parents=True,exist_ok=True)
        if a.stage=='extract':
            from scripts.analyze_native_coupling import extract
            extract(a.data,a.output,corpus=RUN)
        elif a.stage=='scatter':
            for arm,label in zip(ARMS,(r'sensor-noise SD $\eta$','common-mode SD','internal coupling $\\varepsilon$')):
                snr.scatter(a.output,arm,'nuisance',label,f'component-scatter-{arm}')
        elif a.stage=='figure':
            import pandas as pd
            snr.figure(pd.read_csv(a.output/'scores.csv'),a.output,**names)
        else:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):
                snr.analyze(a.data,a.output,'nuisance',**names,qualification=f'{a.size} nuisance arms on fresh seeds. Arms, primary readout and '
                    'scores were fixed before outcomes; internal-coupling is a declared boundary test; supervised mean readouts are information ceilings.')
