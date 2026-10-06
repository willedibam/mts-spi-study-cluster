"""M=48 confirmation of the 2:1 locking sweep under recording-specific nuisances; protocol fixed before p90.

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
from pathlib import Path
import numpy as np
from scripts.cross_frequency_locking import ROOT,EPS,simulate
from scripts import cross_frequency_locking_snr as snr

RUN='cross-frequency-locking-confirm-261006'
DATA=ROOT/'data/order-parameter-inference'/RUN
OUT=ROOT/'results/order-parameter-inference'/RUN
N=16
ARMS=('random-noise','common-mode','internal-coupling')
FIRST_SEED={'random-noise':300,'common-mode':400,'internal-coupling':500}
NOISE_SEED,PYSPI_SEED=261094,261095


def record(task):
    arm,gamma,seed=task;eta,common,eps=.5,0.,EPS
    rng=np.random.default_rng(np.random.SeedSequence([NOISE_SEED,ARMS.index(arm),int(round(gamma*1000)),seed]))
    level=float(rng.uniform())
    if arm==ARMS[0]:eta=.3+.6*level
    if arm==ARMS[1]:common=.8*level
    if arm==ARMS[2]:eps=.15+.45*level
    raw,truth=simulate(gamma,seed,n=N,eps=eps)
    x,truth=snr.observe(raw,truth,rng,eta,common)
    truth.update(common=common,eps=eps,nuisance=dict(zip(ARMS,(eta,common,eps)))[arm])
    return x,truth


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract','analyze','figure'])
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    names=dict(stem='confirm-comparison',heading='2:1 locking under recording-specific nuisances')
    if a.stage=='prepare':snr.prepare(a.workers,RUN,ARMS,FIRST_SEED,record,N,PYSPI_SEED,__file__)
    else:
        a.output.mkdir(parents=True,exist_ok=True)
        if a.stage=='extract':
            from scripts.analyze_native_coupling import extract
            extract(a.data,a.output,corpus=RUN)
        elif a.stage=='figure':
            import pandas as pd
            snr.figure(pd.read_csv(a.output/'scores.csv'),a.output,**names)
        else:
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=4):
                snr.analyze(a.data,a.output,'nuisance',**names,qualification='M=48 confirmation on fresh seeds. Arms, primary readout and '
                    'scores were fixed before outcomes; internal-coupling is a declared boundary test; supervised mean readouts are information ceilings.')
