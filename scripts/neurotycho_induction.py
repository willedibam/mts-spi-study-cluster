"""Exploratory: time-resolved NeuroTycho anaesthetic induction as a real regime change for SPI-SPI.

Uses the staged KTMD and propofol sessions and the pilot's montage and preprocessing unchanged
(16 bipolar channels, 8 s at 250 Hz, 28 s filter context). Unlike the pilot's stable-state windows,
windows here tile the injection session: before injection, induction, the labelled anaesthetized
interval and, where recorded, emergence and recovery; plus the separate awake-eyes-closed session.
There is no measured control or per-window order parameter: truth is the event markers. Fitting is
target-blind and leaves the evaluated animal out.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import spearmanr
from src.neurotycho_pilot import preprocess

ROOT=Path(__file__).resolve().parents[1]
RUN='neurotycho-induction-261006'
DATA=ROOT/'data/order-parameter-inference'/RUN
OUT=ROOT/'results/order-parameter-inference'/RUN
REMOTE='/scratch/ql44/we2614/mts-spi-study/order-parameter-inference/'+RUN
RAW=ROOT/'data/representation/neurotycho/inputs'
PILOT=ROOT/'results/representation/neurotycho'
SOURCES=[(RAW/'neurotycho_source_260910',PILOT/'neurotycho_source_pilot_260910/float64/all-source','ktmd'),
         (RAW/'neurotycho_target_260910',PILOT/'neurotycho_target_pilot_260910','propofol')]
STRIDE=dict(awake=40000,injection=20000,after=60000)
PYSPI_SEED=261098


def events(session):
    c=loadmat(session/'Condition.mat',simplify_cells=True)
    labels=[str(x) for x in np.atleast_1d(c['ConditionLabel'])];index=np.atleast_1d(c['ConditionIndex']).astype(int)-1
    return list(zip(labels,index.tolist()))


def windows(marks,n):
    """Window starts (1 kHz samples) and phases for one session of n samples."""
    first=dict();[first.setdefault(k,v) for k,v in marks];out=[];feasible=lambda s:s>=10000 and s+18000<=n
    if 'AnestheticInjection' in first:
        t0=first['AnestheticInjection'];a0=first.get('Anesthetized-Start',n);a1=first.get('Anesthetized-End',n)
        r0=first.get('RecoveryEyesClosed-Start',n);start=10000
        while feasible(start):
            mid=start+4000
            phase='pre' if start+8000<=t0 else 'induction' if mid<a0 else 'anaesthetized' if mid<a1 else 'emergence' if mid<r0 else 'recovery'
            out.append((start,phase,(mid-t0)/1000));start+=STRIDE['injection'] if mid<a1 else STRIDE['after']
    elif 'AwakeEyesClosed-Start' in first:
        for start in range(first['AwakeEyesClosed-Start']+30000,first['AwakeEyesClosed-End']-38000,STRIDE['awake']):
            if feasible(start):out.append((start,'awake',np.nan))
    return out


def prepare():
    import yaml
    from scripts.spi_baseline_exploration import sha
    if (DATA/'manifest.json').exists():raise FileExistsError(DATA)
    DATA.mkdir(parents=True,exist_ok=True);rows=[];raw={};rejected=0
    for root,meta_root,agent in SOURCES:
        for archive in sorted(p for p in root.iterdir() if p.is_dir()):
            meta=json.loads((meta_root/f'{archive.name}.json').read_text());pairs=np.array(meta['montage']['pairs'])
            channels=sorted(set(pairs.ravel().tolist()));local=np.vectorize(channels.index)(pairs)
            for session in sorted(archive.glob('Session*')):
                if not (session/'Condition.mat').exists() or not (session/f'ECoG_ch{channels[0]}.mat').exists():continue
                marks=events(session)
                if not windows(marks,10**12):continue
                signal=np.stack([np.ravel(next(v for k,v in loadmat(session/f'ECoG_ch{c}.mat').items() if not k.startswith('__'))).astype(float)
                                 for c in channels])
                for start,phase,time in windows(marks,signal.shape[1]):
                    x,quality=preprocess(signal[:,start-10000:start+18000],local)
                    if x is None:rejected+=1;continue
                    x=(x-x.mean(1,keepdims=True))/x.std(1,keepdims=True)
                    name=f'{archive.name[:8]}-{agent}-{meta["animal"]}-{session.name}-{start:08d}';raw[name]=x
                    r=np.corrcoef(x);off=~np.eye(16,dtype=bool)
                    rows.append(dict(row_id=name,corpus_index=len(rows),M=16,N=16,T=2000,seed=0,instance=len(rows),block=archive.name,label=agent,
                        system=agent,control=None if np.isnan(time) else float(time),role='evaluation',animal=meta['animal'],archive=archive.name,
                        session=session.name,agent=agent,phase=phase,time=None if np.isnan(time) else float(time),start=int(start),
                        mean_abs_r=float(abs(r[off]).mean()),mean_r=float(r[off].mean())))
                print(archive.name,session.name,[m for m in marks],len(rows),flush=True)
    np.savez_compressed(DATA/'observations.npz',**raw,__dataset_names__=np.array(list(raw)),
        __labels_json__=np.array([json.dumps([r['agent'],r['animal'],r['phase']]) for r in rows]),__shapes__=np.array([[16,2000]]*len(rows)),
        __axis_order__=np.array(['process','observation']))
    manifest=dict(rows=rows,corpus=RUN,archive_sha256=sha(DATA/'observations.npz'),generator_sha256=sha(__file__),rejected_windows=rejected,
        analysis_scope='Exploratory real-data attempt; truth is event markers only. Leave-one-animal-out target-blind fitting.')
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config=dict(name=RUN,source=dict(format='named-npz-v1',archive=REMOTE+'/observations.npz',sha256=manifest['archive_sha256'],
        axis_order=['process','observation']),base_output_dir=REMOTE+'/mpis',pyspi_config='configs/pyspi/benchmarked_p90.yaml',
        normalise=False,random_seed=PYSPI_SEED)
    (DATA/'corpus.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    total=len(rows);smoke=[1,total];rest=[i for i in range(1,total+1) if i not in smoke]
    for name,indices in dict(smoke=smoke,rest=rest).items():(DATA/f'{name}-indices.txt').write_text(''.join(f'{i}\n' for i in indices))
    frame=pd.DataFrame(rows);print(frame.groupby(['agent','animal','archive']).phase.value_counts().unstack(fill_value=0).to_string())
    print(total,'rejected',rejected,manifest['archive_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','extract'])
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:
        a.output.mkdir(parents=True,exist_ok=True)
        from scripts.analyze_native_coupling import extract
        extract(a.data,a.output,corpus=RUN)
