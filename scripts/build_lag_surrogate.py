"""Gaussian lag switching versus a covariance/cross-spectrum matched surrogate."""
import json
from pathlib import Path
import numpy as np
import yaml
from scripts.spi_baseline_exploration import ROOT, sha

RUN = 'lag-surrogate-261004'
DATA = ROOT/'data/representation'/RUN
OUT = ROOT/'results/representation'/RUN


def pair(rng, strength=1., m=16, t=1000, lag=5):
    assert m % 2 == 0 and t % (2*lag) == 0
    weights = strength*np.linspace(.35, .85, m//2)
    rng.shuffle(weights)
    eps = rng.normal(size=(m,t))
    x = eps.copy()
    permutation = np.roll(np.arange(t).reshape(-1,2*lag),lag,axis=1).ravel()
    assert len(np.unique(permutation)) == t and np.all(permutation != np.arange(t))
    for j, a in enumerate(weights):
        x[2*j+1] = a*eps[2*j,permutation] + np.sqrt(1-a*a)*eps[2*j+1]
    x = x[rng.permutation(m)]
    x -= x.mean(axis=1,keepdims=True)
    x /= x.std(axis=1,keepdims=True)
    f = np.fft.rfft(x)
    phase = np.exp(1j*rng.uniform(-np.pi,np.pi,f.shape[1]))
    phase[0] = phase[-1] = 1
    y = np.fft.irfft(f*phase,n=t)
    fy = np.fft.rfft(y)
    covariance_error = float(np.max(abs(x@x.T/t-y@y.T/t)))
    spectrum_error = float(np.max(abs(f[:,None]*f.conj()[None,:]-fy[:,None]*fy.conj()[None,:])) / np.max(abs(f))**2)
    assert covariance_error < 1e-12 and spectrum_error < 1e-12
    return (x,y), dict(covariance_max_error=covariance_error,cross_spectrum_relative_error=spectrum_error)


def build():
    if (DATA/'manifest.json').exists():
        raise FileExistsError('Immutable bank already exists')
    DATA.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    arrays, records, audits = {}, [], []
    # The original cheap scout used exactly this development RNG stream.
    for panel, n, seed, strength in [('matched',48,261013,1.),('null',16,261014,0.),('held',32,261015,1.)]:
        rng=np.random.default_rng(seed)
        for block in range(n):
            data, audit=pair(rng,strength)
            audits.append(dict(panel=panel,block=block,**audit))
            for label,x in zip(['alternating-lag','shared-phase'],data):
                name=f'{panel}-{label}-block-{block:02d}'
                arrays[name]=x
                cov=x@x.T/x.shape[1]; off=cov[~np.eye(len(x),dtype=bool)]
                records.append(dict(row_id=name,label=label,block=block,instance=block,M=16,T=1000,
                    corpus_index=len(records),panel='matched' if panel=='held' else panel,
                    role='evaluation' if panel=='held' else 'development',
                    development_part='held' if panel=='held' else 'control' if panel=='null' else 'train' if block<32 else 'validation',
                    mean_covariance=float(off.mean()),mean_abs_Pearson=float(abs(off).mean()),seed=seed,strength=strength))
    np.savez_compressed(DATA/'observations.npz',**arrays,
        __dataset_names__=np.array(list(arrays)),__labels_json__=np.array([json.dumps([r['label']]) for r in records]),
        __shapes__=np.array([[16,1000]]*len(records)),__axis_order__=np.array(['process','observation']))
    manifest=dict(rows=records,archive_sha256=sha(DATA/'observations.npz'),generator_sha256=sha(__file__),
        protocol_sha256=sha(ROOT/f'configs/analysis/{RUN}.yaml'),invariants=audits)
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config=dict(name=RUN,source=dict(format='named-npz-v1',archive=f'/scratch/ql44/we2614/mts-spi-study/representation/synthetic/{RUN}/observations.npz',sha256=manifest['archive_sha256'],axis_order=['process','observation']),
        base_output_dir=f'/scratch/ql44/we2614/mts-spi-study/representation/synthetic/{RUN}/mpis',pyspi_config='configs/pyspi/benchmarked_p90.yaml',normalise=False,random_seed=261016)
    (ROOT/f'configs/external/{RUN}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    print(len(records),manifest['archive_sha256'])

if __name__=='__main__':
    build()
