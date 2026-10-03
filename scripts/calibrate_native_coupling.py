"""Match total off-diagonal Jacobian gain in existing VAR and CML generators."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scripts.spi_baseline_exploration import ROOT, sha
from src.generators import generate_varma, generate_cml_logistic

OUT=ROOT/'results/representation/native-gain-261003'
DATA=ROOT/'data/representation/native-gain-261003'
CASES=[('VAR',.2),('VAR',.7)]+[('CML',a) for a in [1.45,1.69,1.7522,1.895,2.0]]
TARGET=.20


def generate(family,parameter,g,seed,t=1000):
    rng=np.random.default_rng(seed)
    if family=='VAR':
        assert parameter+g < .98
        x=generate_varma(M=16,T=t,phi=parameter,coupling=g/2,ma_phi=0,ma_coupling=0,
                        noise_std=.1,transients=2000,zscore=False,rng=rng)
        effect=g/2*(np.roll(x,1,axis=1)+np.roll(x,-1,axis=1))
    else:
        x,full=generate_cml_logistic(M=16,T=t,alpha=parameter,eps=g,transients=2000,
                                    sample_every=1,zscore=False,rng=rng,return_full_lattice=True)
        f=1-parameter*full**2
        delta=g*((np.roll(f,1,axis=1)+np.roll(f,-1,axis=1))/2-f)
        start=(full.shape[1]-16)//2;effect=delta[:,start:start+16]
    # Mean row sum of absolute off-diagonal Jacobian entries.
    gain=float(g) if family=='VAR' else float(g*parameter*np.mean(np.abs(np.roll(full,1,axis=1)[:,start:start+16])+np.abs(np.roll(full,-1,axis=1)[:,start:start+16])))
    variance=np.var(x,axis=0).sum()
    strength=float(np.sqrt(np.mean(np.sum(effect**2,axis=1))/variance)) if variance>0 else np.nan
    mask=~np.eye(16,dtype=bool);r=np.corrcoef(x.T)
    diagnostics=dict(strength=gain,realized_effect=strength,mean_abs_Pearson=float(np.mean(abs(r[mask]))),
                     min_channel_SD=float(np.std(x,axis=0).min()),mean_channel_SD=float(np.std(x,axis=0).mean()))
    return x,diagnostics


def scout():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for family,parameter in CASES:
        upper=min(.6,.96-parameter) if family=='VAR' else .6
        for g in np.linspace(0,upper,13):
            for instance in range(3):
                _,d=generate(family,parameter,float(g),26100300+instance)
                rows.append(dict(family=family,parameter=parameter,g=g,instance=instance,**d))
        print(family,parameter,'completed',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'calibration-scout.csv',index=False)


def calibrate(family,parameter,seed):
    if family=='VAR':
        x,d=generate(family,parameter,TARGET,seed)
        return TARGET,x,d,1
    upper=min(.6,.96-parameter) if family=='VAR' else .2
    tried=[]
    def evaluate(g):
        x,d=generate(family,parameter,float(g),seed)
        tried.append((float(g),x,d))
        return d['strength']-TARGET
    left=0.;fleft=evaluate(left)
    for right in np.linspace(0,upper,9)[1:]:
        fright=evaluate(right)
        if fleft*fright<=0:break
        left,fleft=right,fright
    else:raise ValueError(f'No target bracket: {family} {parameter} seed {seed}')
    for _ in range(24):
        best=min(tried,key=lambda v:abs(v[2]['strength']-TARGET))
        if abs(best[2]['strength']-TARGET)<=.002:return (*best,len(tried))
        middle=(left+right)/2;fm=evaluate(middle)
        if fleft*fm<=0:right=middle
        else:left,fleft=middle,fm
    best=min(tried,key=lambda v:abs(v[2]['strength']-TARGET))
    return (*best,len(tried))


def build():
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (DATA/'manifest.json').exists():raise FileExistsError('Existing immutable calibration bank')
    arrays,rows={},[]
    for instance in range(24):
        for family,parameter in CASES:
            seed=261003100+instance
            g,x,d,attempts=calibrate(family,parameter,seed)
            label=f'{family}-{parameter:g}';name=f'{label}-I{instance:02d}'
            assert abs(d['strength']-TARGET)<=.002,(name,g,d)
            arrays[name]=x.T
            rows.append(dict(row_id=name,label=label,family=family,parameter=parameter,g=g,
                block=instance,seed=seed,role='development' if instance<12 else 'evaluation',
                corpus_index=len(rows),M=16,T=1000,calibration_evaluations=attempts,**d))
        print('calibrated block',instance,flush=True)
    np.savez_compressed(DATA/'observations.npz',**arrays,
        __dataset_names__=np.array([r['row_id'] for r in rows]),
        __labels_json__=np.array([json.dumps([r['label']]) for r in rows]),
        __shapes__=np.array([[16,1000]]*len(rows)),__axis_order__=np.array(['process','observation']))
    manifest=dict(rows=rows,archive_sha256=sha(DATA/'observations.npz'),script_sha256=sha(Path(__file__)),
                  protocol_sha256=sha(ROOT/'configs/analysis/native-coupling-261003.yaml'))
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pd.DataFrame(rows).drop(columns=['row_id']).to_csv(OUT/'calibration.csv',index=False)
    prepare_observations()


def prepare_observations():
    """Remove arbitrary global amplitude without changing the strength ratio."""
    p=DATA/'observations.npz';native=DATA/'native-observations.npz'
    assert not native.exists()
    p.rename(native)
    manifest=json.loads((DATA/'manifest.json').read_text())
    with np.load(native) as a:
        arrays={k:a[k] for k in a.files}
    for row in manifest['rows']:
        x=arrays[row['row_id']];center=x.mean(axis=1,keepdims=True)
        scale=float(np.sqrt(np.mean(np.var(x,axis=1))))
        arrays[row['row_id']]=(x-center)/scale
        row['observation_center']=center[:,0].tolist();row['observation_scale']=scale
        np.testing.assert_allclose(np.mean(np.var(arrays[row['row_id']],axis=1)),1,atol=1e-12)
    np.savez_compressed(p,**arrays)
    manifest['native_archive_sha256']=sha(native);manifest['archive_sha256']=sha(p)
    manifest['observation_transform']='per_channel_center_one_global_RMS_SD_scale; Jacobian_gain_unchanged; no_MPI_normalization'
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['scout','build']);args=p.parse_args()
    {'scout':scout,'build':build}[args.stage]()
