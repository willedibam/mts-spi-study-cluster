"""Six band assignments: fixed individual-band strengths, changing correspondence."""
from pathlib import Path
from itertools import permutations
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from scripts.spi_baseline_exploration import ROOT, project_features, sha

DATA = ROOT/'data/representation/band-organization-fixed-strength-261003'
OUT = ROOT/'results/representation/band-organization-fixed-strength-261003'
BANDS = ((.035,.095),(.17,.23),(.33,.39))
CLASSES = tuple(''.join(p) for p in permutations('ABC'))


def covariance_layers(strength):
    a = np.repeat(np.arange(4),4)
    b = np.tile(np.arange(4),4)
    c = a.copy()
    c[[1,4]] = c[[4,1]]
    c[[10,13]] = c[[13,10]]
    return np.array([(1-strength)*np.eye(16)+strength*(g[:,None]==g[None,:]) for g in (a,b,c)])


def frequency_weights(t):
    f = np.fft.rfftfreq(t)
    weights=[]
    for lo,hi in BANDS:
        w=np.zeros(len(f));take=(f>lo)&(f<hi)
        w[take]=np.sin(np.pi*(f[take]-lo)/(hi-lo))
        w*=np.sqrt(t/(2*np.sum(w*w)))
        weights.append(w)
    return np.array(weights)


def simulate(label, block, t=1000):
    nuisance=np.random.default_rng(np.random.SeedSequence([261003,block,0]))
    strength=.55;order=nuisance.permutation(16)
    layers=covariance_layers(strength)
    rng=np.random.default_rng(np.random.SeedSequence([261003,block,1,CLASSES.index(label)]))
    weights=frequency_weights(t);coefficients=np.zeros((t//2+1,16),complex)
    for w,name in zip(weights,label):
        noise=(rng.normal(size=coefficients.shape)+1j*rng.normal(size=coefficients.shape))/np.sqrt(2)
        coefficients+=w[:,None]*(noise@np.linalg.cholesky(layers['ABC'.index(name)]).T)/np.sqrt(3)
    x=np.fft.irfft(coefficients,n=t,axis=0,norm='ortho')[:,order]
    return x,dict(strength=strength,sensor_order=order.tolist())


def marginal(a):
    v=a[~np.eye(len(a),dtype=bool)]
    return np.r_[v.mean(),v.std(),np.quantile(v,[.1,.25,.5,.75,.9])]


def direct_features(x):
    t,m=x.shape;mask=~np.eye(m,dtype=bool);ft=np.fft.rfft(x,axis=0,norm='ortho');f=np.fft.rfftfreq(t)
    profiles=[]
    for lo,hi in BANDS:
        select=(f>lo)&(f<hi)
        band=np.fft.irfft(ft*select[:,None],n=t,axis=0,norm='ortho')
        profiles.append(np.cov(band,rowvar=False,bias=True))
    edge=np.array([p[mask] for p in profiles])
    spectra=np.abs(ft)**2
    raw=np.cov(x,rowvar=False,bias=True)
    return dict(band_marginals=np.concatenate([marginal(p) for p in profiles]),
        band_z=np.corrcoef(edge)[np.triu_indices(3,1)],
        raw_covariance=marginal(raw),
        raw_Pearson=marginal(np.corrcoef(x.T)),
        spectra=np.concatenate([np.mean(spectra[(f>lo)&(f<hi)],axis=0)[None,:] for lo,hi in BANDS]).mean(axis=1))


def build():
    if (DATA/'manifest.json').exists():raise FileExistsError('Preserve the existing immutable pilot.')
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    edges=~np.eye(16,dtype=bool);weights=frequency_weights(1000)
    np.testing.assert_allclose(2*np.sum(weights**2,axis=1)/1000,1,atol=1e-14)
    assert np.all((weights>0).sum(axis=0)<=1)
    signatures=[]
    for label in CLASSES:
        layers=covariance_layers(.55)[['ABC'.index(c) for c in label]]
        assert np.linalg.eigvalsh(layers).min()>0
        for layer in layers:np.testing.assert_allclose(np.sort(layer[edges]),np.sort(layers[0][edges]))
        np.testing.assert_allclose(layers.mean(axis=0),covariance_layers(.55).mean(axis=0))
        signatures.append(np.corrcoef(layers[:,edges])[np.triu_indices(3,1)])
    assert len(np.unique(np.round(signatures,12),axis=0))==6
    rows,arrays,features=[],{},{}
    for block in range(24):
        for label in CLASSES:
            name=f'{label}-block-{block:02d}'
            x,meta=simulate(label,block)
            arrays[name]=x.T
            row=dict(row_id=name,label=label,block=block,seed=block,role='development' if block<12 else 'evaluation',
                     corpus_index=len(rows),M=16,T=1000,**meta)
            rows.append(row)
            for k,v in direct_features(x).items():features.setdefault(k,[]).append(v)
    np.savez_compressed(DATA/'observations.npz',**arrays,
        __dataset_names__=np.array([r['row_id'] for r in rows]),
        __labels_json__=np.array([json.dumps([r['label']]) for r in rows]),
        __shapes__=np.array([[16,1000] for r in rows]),
        __axis_order__=np.array(['process','observation']))
    np.savez_compressed(DATA/'direct-features.npz',row_id=np.array([r['row_id'] for r in rows]),**{k:np.array(v) for k,v in features.items()})
    manifest=dict(rows=rows,population_signatures=dict(zip(CLASSES,np.array(signatures).tolist())),
                  archive_sha256=sha(DATA/'observations.npz'),script_sha256=sha(Path(__file__)),
                  protocol_sha256=sha(ROOT/'configs/analysis/band-organization-261003.yaml'),population_identities_verified=True)
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    analyze_direct()


def analyze_direct():
    # Pilot gate examines development data only. Hold out the last four development
    # blocks; the twelve evaluation blocks remain sealed until p90/readouts freeze.
    rows=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows'])
    train=(rows.block<8).to_numpy();test=((rows.block>=8)&(rows.block<12)).to_numpy()
    scores=[]
    with np.load(DATA/'direct-features.npz') as a:
        for name in a.files:
            if name=='row_id':continue
            _,d,h=project_features(a[name][train],a[name][test],dimensions=20)
            model=LogisticRegression(C=1,max_iter=3000).fit(d,rows.label[train])
            scores.append(dict(method=name,development_holdout_BA=balanced_accuracy_score(rows.label[test],model.predict(h))))
    pd.DataFrame(scores).to_csv(OUT/'direct-development-gate.csv',index=False)
    print(pd.DataFrame(scores).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','direct']);args=p.parse_args()
    with threadpool_limits(limits=4):
        {'build':build,'direct':analyze_direct}[args.stage]()
