"""Proof classes observed at controlled coupling strength: is the mean-SPI baseline strength, and SPI-SPI character?

Eight coupled classes (two VARs, Kuramoto, wave, four CML regimes) and two independent-noise references, M=16, T=1000.
Every arm observes the SAME dynamical realization per (class, instance) through white Gaussian sensor noise of SD eta,
in signal-SD units; arms differ only in how eta is set per recording:
  native     eta=0.
  mild       eta~U[.3,.9], independent of class: recording quality varies (the construction that held for 2:1 locking).
  wide       Pearson attenuation u=1/(1+eta^2)~U[.25,.95], independent of class.
  equalised  eta solves mean|r|=.14: strength normalised to one value in every class.
  matched    eta solves mean|r|=s, s~U[.08,.20]: strength normalised to one DISTRIBUTION in every class and randomised.
Sensor noise only attenuates, so a recording natively weaker than its target is left as generated (capped; recorded).
Gaussian and Cauchy noise have no coupling to attenuate (same-family observation noise leaves their law unchanged);
one set is generated and joins every arm.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
RUN='proof-strength-nuisance-261006'
DATA=ROOT/'data/proof'/RUN
OUT=ROOT/'results/proof'/RUN
REMOTE='/scratch/ql44/we2614/mts-spi-study/proof/'+RUN
M,T,INSTANCES=16,1000,40
SEED,PYSPI_SEED=261101,261102
ARMS=('native','mild','wide','equalised','matched')
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
    arm,name,instance=task;x=dynamics(name,instance)
    if name in NOISE:return x,dict(eta=0.,target=None,capped=False,native_abs_r=strength(x),mean_abs_r=strength(x),
                                   mean_abs_spearman=float(abs(spearmanr(x.T).statistic[OFF]).mean()))
    return observe(x,arm,np.random.default_rng(np.random.SeedSequence([SEED+1,ARMS.index(arm),list(CLASSES).index(name),instance])))


def tasks():
    coupled=[(arm,name,i) for arm in ARMS for name in CLASSES if name not in NOISE for i in range(INSTANCES)]
    return coupled+[('all',name,i) for name in NOISE for i in range(INSTANCES)]


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


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract'])
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.stage=='prepare':prepare(a.workers)
    else:
        from scripts.analyze_native_coupling import extract
        a.output.mkdir(parents=True,exist_ok=True);extract(a.data,a.output,corpus=RUN)
