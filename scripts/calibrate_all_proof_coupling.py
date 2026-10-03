"""Extend the native-gain pilot to every historical proof class; reuse its inputs."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import yaml
from scripts import calibrate_native_coupling as pilot
from scripts.spi_baseline_exploration import ROOT, sha
from src.generators import generate_kuramoto, generate_wave_1d

DATA=ROOT/'data/representation/proof-strength-all-261003'
OUT=ROOT/'results/representation/proof-strength-all-261003'
TARGET=.2
# Preserve actual legacy self coefficients after its stability rescaling.
LEGACY_VAR=[(.95,.4),(.2,.4),(.2,.8)]
EXTRA=[dict(label=f'VAR-legacy-{p:g}-{c:g}',family='VAR',parameter=.98*p/(p+2*c)) for p,c in LEGACY_VAR]
EXTRA += [dict(label='CML-1.85',family='CML',parameter=1.85),
          dict(label='Kuramoto-fast',family='Kuramoto',parameter=3.),
          dict(label='Kuramoto-slow',family='Kuramoto',parameter=.025),
          dict(label='Gaussian-noise',family='noise',parameter=0.),
          dict(label='Cauchy-noise',family='noise',parameter=1.),
          dict(label='Wave',family='wave',parameter=10.)]


def kuramoto_gain(theta,K,dt=.00625):
    # Degree-normalized all-to-all phase update; diagonal is not cross-channel.
    diff=theta[:,:,None]-theta[:,None,:]
    values=np.abs(np.cos(diff));values[:,np.arange(theta.shape[1]),np.arange(theta.shape[1])]=0
    return float(abs(K)*dt*np.mean(values.sum(axis=2))/(theta.shape[1]-1))


def simulate(case,g,seed,t=1000):
    family,p=case['family'],case['parameter']
    if family in ('VAR','CML'):
        return pilot.generate(family,p,g,seed,t=t)
    rng=np.random.default_rng(seed)
    if family=='Kuramoto':
        theta=generate_kuramoto(M=16,T=t,dt=.00625,K=-g,omega_mean=p,
            omega_std=1.73205 if p==3 else .01443,eta=0,output='phase',
            connectivity='all-to-all',transients=2000,zscore=False,rng=rng)
        x=np.sin(theta);gain=kuramoto_gain(theta,-g)
    elif family=='wave':
        # Fix the historical c=10 timestep, then vary physical wave speed.
        dt=.2/(16*10)
        x=generate_wave_1d(M=16,T=t,c=g,dt=dt,n_modes=5,ic_decay=1.5,
                           noise_std=.01,zscore=False,rng=rng)
        gain=2*(g*dt*16)**2
    elif family=='noise':
        x=rng.normal(size=(t,16)) if p==0 else rng.standard_cauchy(size=(t,16));gain=0.
    else:raise ValueError(family)
    mask=~np.eye(16,dtype=bool)
    return x,dict(strength=gain,mean_abs_Pearson=float(np.mean(abs(np.corrcoef(x.T)[mask]))),
                  min_channel_SD=float(np.std(x,axis=0).min()),mean_channel_SD=float(np.std(x,axis=0).mean()))


def calibrate(case,seed):
    family=case['family']
    if family in ('VAR','CML'):return pilot.calibrate(family,case['parameter'],seed)
    if family in ('noise','wave'):
        g=0. if family=='noise' else 10*np.sqrt(TARGET/.08)
        x,d=simulate(case,g,seed);return g,x,d,1
    tried=[]
    def evaluate(g):
        x,d=simulate(case,g,seed);tried.append((float(g),x,d));return d['strength']-TARGET
    left=0.;fl=evaluate(left)
    for right in np.arange(10.,91.,10.):
        fr=evaluate(right)
        if fl*fr<=0:break
        left,fl=right,fr
    else:raise ValueError(f'No phase-gain bracket: {case} seed={seed}')
    for _ in range(24):
        best=min(tried,key=lambda item:abs(item[2]['strength']-TARGET))
        if abs(best[2]['strength']-TARGET)<=.002:return (*best,len(tried))
        middle=(left+right)/2;fm=evaluate(middle)
        if fl*fm<=0:right=middle
        else:left,fl=middle,fm
    raise ValueError(f'Phase gain did not converge: {case} seed={seed}')


def inventory():
    old=yaml.safe_load((ROOT/'configs/generate/embeddings/cross-mt-confirmation-260824.yaml').read_text())
    mapping={}
    for entry,case in zip(old['mts_classes'][:3],EXTRA[:3]):mapping[entry['name']]=case['label']
    mapping.update({'kuramoto_omega-fast':'Kuramoto-fast','kuramoto_omega-slow':'Kuramoto-slow',
        'cauchy-noise':'Cauchy-noise','gaussian-noise':'Gaussian-noise','wave-1d':'Wave'})
    for entry in old['mts_classes'][8:]:mapping[entry['name']]=f"CML-{entry['base_params']['alpha']:g}"
    rows=[dict(source='historical14',original=name,normalized=label) for name,label in mapping.items()]
    fresh=yaml.safe_load((ROOT/'configs/generate/embeddings/proof-p90-260924.yaml').read_text())
    for entry in fresh['mts_classes'][:3]:
        rows.append(dict(source='fresh_VAR',original=entry['name'],normalized=f"VAR-{entry['base_params']['phi']:g}"))
    return pd.DataFrame(rows)


def build():
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if (DATA/'manifest.json').exists():raise FileExistsError('Immutable input bank exists')
    mapping=inventory();mapping.to_csv(OUT/'class-inventory.csv',index=False)
    native,observations,rows={},{},[]
    for block in range(24):
        for case in EXTRA:
            seed=261003100+block;g,x,d,n=calibrate(case,seed)
            is_null=case['family']=='noise';target=0. if is_null else TARGET
            assert abs(d['strength']-target)<=.002
            row_id=f"{case['label']}-I{block:02d}";x=x.T
            center=x.mean(axis=1,keepdims=True);scale=float(np.sqrt(np.mean(np.var(x,axis=1))))
            assert scale>0 and np.isfinite(x).all()
            native[row_id]=x;observations[row_id]=(x-center)/scale
            rows.append(dict(**case,row_id=row_id,block=block,seed=seed,g=g,role='development' if block<12 else 'evaluation',
                corpus_index=len(rows),M=16,T=1000,calibration_evaluations=n,target=target,
                gain_coordinates='latent_phase' if case['family']=='Kuramoto' else 'native_observed_state',
                observation_center=center[:,0].tolist(),observation_scale=scale,**d))
        print('all-proof block',block,'complete',flush=True)
    metadata=dict(__dataset_names__=np.array([r['row_id'] for r in rows]),
        __labels_json__=np.array([json.dumps([r['label']]) for r in rows]),
        __shapes__=np.array([[16,1000]]*len(rows)),__axis_order__=np.array(['process','observation']))
    for name,arrays in [('native-observations',native),('observations',observations)]:
        np.savez_compressed(DATA/f'{name}.npz',**arrays,**metadata)
    manifest=dict(rows=rows,archive_sha256=sha(DATA/'observations.npz'),native_archive_sha256=sha(DATA/'native-observations.npz'),
        script_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'configs/analysis/proof-strength-all-261003.yaml'))
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    old=json.loads((pilot.DATA/'manifest.json').read_text())
    combined=[dict(r,source_bank='native-gain-261003',target=.2,gain_coordinates='native_observed_state') for r in old['rows']]
    combined += [dict(r,source_bank='proof-strength-all-261003') for r in rows]
    for r in combined:
        r['historical14']=r['label'] in set(mapping.query("source=='historical14'").normalized)
        r['coupled']=r['family']!='noise'
    (OUT/'combined-manifest.json').write_text(json.dumps(dict(rows=combined,
        source_manifests={bank:sha(ROOT/f'data/representation/{bank}/manifest.json') for bank in ['native-gain-261003','proof-strength-all-261003']}),indent=2)+'\n')
    pd.DataFrame(combined).drop(columns=['observation_center']).to_csv(OUT/'calibration.csv',index=False)
    external=yaml.safe_load((ROOT/'configs/external/native-gain-261003.yaml').read_text())
    external['name']='proof-strength-all-261003'
    external['source']['archive']=external['source']['archive'].replace('native-gain-261003',external['name'])
    external['source']['sha256']=manifest['archive_sha256']
    external['base_output_dir']=external['base_output_dir'].replace('native-gain-261003',external['name'])
    (ROOT/'configs/external/proof-strength-all-261003.yaml').write_text(yaml.safe_dump(external,sort_keys=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['build','inventory']);args=parser.parse_args()
    if args.stage=='build':build()
    else:print(inventory().to_string(index=False))
