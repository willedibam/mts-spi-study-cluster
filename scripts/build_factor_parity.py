"""Two Gaussian factor models with identical population coupling magnitudes."""
from pathlib import Path
import json
import numpy as np
import yaml
from scipy.stats import rankdata
from scipy.signal import correlate
from scripts.spi_baseline_exploration import ROOT,sha
from scripts.band_organization_experiment import marginal

RUN='factor-parity-261004'
DATA=ROOT/'data/representation'/RUN
OUT=ROOT/'results/representation'/RUN
CLASSES=('linked-signs','unlinked-signs')
STRENGTH=.35
WEIGHTS=np.array([.7,.2,.1])


def covariance(label,strength=STRENGTH):
    indices=[1,2,3] if label==CLASSES[0] else [1,2,4]
    if label not in CLASSES:raise ValueError(label)
    signs=np.array([[(-1.)**((i&k).bit_count()%2) for i in range(16)] for k in indices])
    return (1-strength)*np.eye(16)+strength*np.einsum('k,ki,kj->ij',WEIGHTS,signs,signs)


def simulate(label,block):
    rng=np.random.default_rng(np.random.SeedSequence([261004,block,CLASSES.index(label)]))
    c=covariance(label)
    x=rng.normal(size=(1000,16))@np.linalg.cholesky(c).T
    order=np.random.default_rng(np.random.SeedSequence([261004,block,99])).permutation(16)
    return x[:,order],dict(strength=STRENGTH,weights=WEIGHTS.tolist(),sensor_order=order.tolist())


def direct_features(x):
    m=x.shape[1];mask=~np.eye(m,dtype=bool)
    cov=np.cov(x,rowvar=False,bias=True);r=np.corrcoef(x.T);s=np.corrcoef(rankdata(x,axis=0).T)
    ed=np.sqrt(np.maximum(np.diag(cov)[:,None]+np.diag(cov)[None,:]-2*cov,0))
    mi=-.5*np.log(np.maximum(1-r*r,1e-12));prec=np.linalg.inv(cov)
    z=(x-x.mean(axis=0))/x.std(axis=0)
    t=len(x);q=t//4
    xc=np.array([correlate(z[:,i],z[:,j],mode='full',method='fft')[t-1-q:t+q].max()/t for i in range(m) for j in range(i+1,m)])
    banks=[cov,r,r*r,s,s*s,ed,mi,prec,prec*prec]
    return dict(probe_mean=np.r_[[a[mask].mean() for a in banks],xc.mean()],
        probe_z=np.corrcoef(np.array([a[mask] for a in banks]))[np.triu_indices(len(banks),1)],
        raw_covariance=marginal(cov),raw_Pearson=marginal(r))


def build():
    if (DATA/'manifest.json').exists():raise FileExistsError('Preserve immutable bank')
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    mask=~np.eye(16,dtype=bool);a,b=[covariance(c) for c in CLASSES]
    np.testing.assert_allclose(np.sort(np.abs(a[mask])),np.sort(np.abs(b[mask])))
    np.testing.assert_allclose(np.linalg.eigvalsh(a),np.linalg.eigvalsh(b))
    np.testing.assert_allclose(a.sum(axis=1),b.sum(axis=1))
    np.testing.assert_allclose(np.abs(a-np.eye(16)).sum(axis=1),np.abs(b-np.eye(16)).sum(axis=1))
    assert np.array_equal((a[mask]<0).sum(),(b[mask]<0).sum())
    arrays={};rows=[];features={}
    for block in range(64):
        for label in CLASSES:
            x,meta=simulate(label,block);name=f'{label}-block-{block:02d}';arrays[name]=x.T
            rows.append(dict(row_id=name,label=label,block=block,role='development' if block<32 else 'evaluation',
                development_part='train' if block<24 else 'validation' if block<32 else 'held',
                corpus_index=len(rows),M=16,T=1000,**meta))
            for key,value in direct_features(x).items():features.setdefault(key,[]).append(value)
    np.savez_compressed(DATA/'observations.npz',**arrays,__dataset_names__=np.array([r['row_id'] for r in rows]),
        __labels_json__=np.array([json.dumps([r['label']]) for r in rows]),__shapes__=np.array([[16,1000]]*len(rows)),
        __axis_order__=np.array(['process','observation']))
    np.savez_compressed(DATA/'direct-features.npz',row_id=np.array([r['row_id'] for r in rows]),**features)
    m=dict(rows=rows,archive_sha256=sha(DATA/'observations.npz'),script_sha256=sha(Path(__file__)),
        protocol_sha256=sha(ROOT/f'configs/analysis/{RUN}.yaml'),
        population_equalities=['absolute covariance multiset','mean signed covariance','row total absolute covariance','covariance eigenvalues','negative edge fraction','zero lagged covariance for all nonzero lags','population MPI marginal values for sign-invariant bivariate measures'],
        qualifications=['signed covariance distributions differ','not nonlinear dynamics: both models are Gaussian and temporally iid','finite-sample joint law of MPI means need not agree','sign-sensitive nonlinear and multivariate SPIs require empirical testing'])
    (DATA/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    config=yaml.safe_load((ROOT/'configs/external/band-swap-261004.yaml').read_text())
    for key in ('name','base_output_dir'):config[key]=config[key].replace('band-swap-261004',RUN)
    config['source']['archive']=config['source']['archive'].replace('band-swap-261004',RUN)
    config['source']['sha256']=m['archive_sha256']
    (ROOT/f'configs/external/{RUN}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    print('Generated',len(rows),'records; archive',m['archive_sha256'])


if __name__=='__main__':build()
