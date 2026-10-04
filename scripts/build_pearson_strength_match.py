"""Raw observation controls matching signed/absolute Pearson summaries at T1000."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq
from scripts.spi_baseline_exploration import ROOT,sha
from scripts.calibrate_all_proof_coupling import calibrate

RUN='pearson-strength-matched-261004'
DATA=ROOT/'data/representation'/RUN
OUT=ROOT/'results/representation'/RUN
COUPLED=[dict(label='VAR-0.7',family='VAR',parameter=.7,noise_index=2),
         dict(label='Wave',family='wave',parameter=10.,noise_index=3),
         dict(label='CML-1.895',family='CML',parameter=1.895,noise_index=0),
         dict(label='Kuramoto-fast',family='Kuramoto',parameter=3.,noise_index=1)]
MASK=~np.eye(16,dtype=bool)
SIGNS=np.ones((32768,16))
SIGNS[:,1:]=1-2*((np.arange(32768)[:,None]>>np.arange(15))&1)


def normalize(x):
    x=np.asarray(x,dtype=float)-np.mean(x,axis=1,keepdims=True)
    sd=x.std(axis=1,keepdims=True)
    if np.any(sd<=0) or not np.isfinite(x).all():raise ValueError('Invalid channel')
    return x/sd


def targets(block):
    absolute=np.random.default_rng(np.random.SeedSequence([261007,block,99])).uniform(.065,.075)
    signed=np.random.default_rng(np.random.SeedSequence([261007,block,98])).uniform(.003,.007)
    return signed,absolute


def absolute_mean(x):
    return float(np.abs(np.corrcoef(x)[MASK]).mean())


def attenuate(x,noise,target):
    x=x-x.mean(axis=1,keepdims=True);noise=noise-noise.mean(axis=1,keepdims=True)
    c=x@x.T/x.shape[1];e=noise@noise.T/x.shape[1]
    cross=(x@noise.T+noise@x.T)/x.shape[1]
    def value(s):
        a=c+s*cross+s*s*e;sd=np.sqrt(np.diag(a))
        return np.abs((a/sd[:,None]/sd[None,:])[MASK]).mean()-target
    if value(0)<0 or value(1e6)>0:raise ValueError('Target outside attenuation range; preserve failure')
    high=1.
    while value(high)>0:high*=2
    sigma=brentq(value,0,high,xtol=1e-12)
    return x+sigma*noise,float(sigma)


def match_polarity(x,target):
    x=normalize(x);r=x@x.T/x.shape[1]
    means=(np.einsum('bi,ij,bj->b',SIGNS,r,SIGNS,optimize=True)-16)/240
    index=int(np.argmin(np.abs(means-target)))
    return SIGNS[index,:,None]*x,SIGNS[index].astype(int).tolist()


def noise_pair(block,cauchy):
    rng=np.random.default_rng(np.random.SeedSequence([261008,block,int(cauchy)]))
    draw=rng.standard_cauchy if cauchy else rng.normal
    independent=draw(size=(16,1000));common=draw(size=(1,1000))
    loadings=rng.permutation(np.linspace(.5,1.5,16))[:,None]
    target=targets(block)[1]
    def x(g):return (1-g)*independent+g*loadings*common
    if absolute_mean(x(0))>=target:raise ValueError('Noise input exceeds positive target; preserve failure')
    g=brentq(lambda g:absolute_mean(x(g))-target,0,1,xtol=1e-13)
    return x(g),normalize(independent),float(g)


def build():
    if (DATA/'manifest.json').exists():raise FileExistsError('Immutable bank already exists')
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    source_manifest=ROOT/'results/representation/proof-strength-all-261003/combined-manifest.json'
    source=pd.DataFrame(json.loads(source_manifest.read_text())['rows'])
    archives={bank:np.load(ROOT/f'data/representation/{bank}/observations.npz') for bank in source.source_bank.unique()}
    arrays={};rows=[]
    for block in range(48):
        signed,absolute=targets(block);items=[]
        for case in COUPLED:
            if block<24:
                r=source[(source.label==case['label'])&(source.block==block)].iloc[0]
                x=archives[r.source_bank][r.row_id].copy()
                provenance=dict(source_bank=r.source_bank,source_row=r.row_id,native_gain=float(r.strength),native_parameter=float(r.g))
            else:
                seed=261008100+block;g,x,diagnostics,_=calibrate(case,seed);x=x.T
                x=(x-x.mean(axis=1,keepdims=True))/np.sqrt(np.var(x,axis=1).mean())
                provenance=dict(source_bank='fresh_generator',source_seed=seed,native_gain=float(diagnostics['strength']),native_parameter=float(g))
            rng=np.random.default_rng(np.random.SeedSequence([261007,block,case['noise_index']]))
            x,sigma=attenuate(x,rng.normal(size=x.shape),absolute)
            items.append((case['label'],x,dict(**provenance,noise_sigma=sigma),True))
        controls=[]
        for name,cauchy in [('Gaussian',False),('Cauchy',True)]:
            x,control,g=noise_pair(block,cauchy)
            items.append((name+'-correlated',x,dict(shared_input_weight=g,native_gain=0.,source_bank='fresh_shared_input_noise'),True))
            controls.append((name+'-independent',control,dict(native_gain=0.,source_bank='paired_independent_noise'),False))
        for label,x,provenance,matched in items+controls:
            if matched:x,polarity=match_polarity(x,signed)
            else:polarity=[1]*16
            c=x@x.T/1000;b=float(c[MASK].mean());a=float(np.abs(c[MASK]).mean())
            np.testing.assert_allclose(x.mean(axis=1),0,atol=1e-12)
            np.testing.assert_allclose(np.var(x,axis=1),1,atol=1e-12)
            if matched:
                assert abs(a-absolute)<1e-9,(label,block,a,absolute)
                assert abs(b-signed)<5e-5,(label,block,b,signed)
            name=f'{label}-block-{block:02d}';arrays[name]=x
            rows.append(dict(row_id=name,label=label,block=block,M=16,T=1000,corpus_index=len(rows),
                role='development' if block<24 else 'evaluation',development_part='train' if block<16 else 'validation' if block<24 else 'held',
                panel='matched' if matched else 'independent_control',mean_covariance=b,mean_abs_Pearson=a,
                target_signed=signed if matched else None,target_absolute=absolute if matched else None,polarity=polarity,**provenance))
        print('Generated block',block,flush=True)
    for a in archives.values():a.close()
    np.savez_compressed(DATA/'observations.npz',**arrays,__dataset_names__=np.array([r['row_id'] for r in rows]),
        __labels_json__=np.array([json.dumps([r['label']]) for r in rows]),__shapes__=np.array([[16,1000]]*len(rows)),__axis_order__=np.array(['process','observation']))
    manifest=dict(rows=rows,archive_sha256=sha(DATA/'observations.npz'),script_sha256=sha(Path(__file__)),
        protocol_sha256=sha(ROOT/f'configs/analysis/{RUN}.yaml'),parent_manifest_sha256=sha(source_manifest),
        qualifications=['Observation-space normalization, not universal native coupling equality','Cauchy covariance is finite-sample only','Polarity selected using covariance mean only; no other SPI or z outcomes','Independent controls kept outside matched-class comparison'])
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pd.DataFrame(rows).drop(columns='polarity').to_csv(OUT/'calibration.csv',index=False)
    config=yaml.safe_load((ROOT/'configs/external/factor-parity-replication-261004.yaml').read_text())
    config['name']=RUN
    config['source']['archive']=f'/scratch/ql44/we2614/mts-spi-study/representation/synthetic/{RUN}/observations.npz'
    config['source']['sha256']=manifest['archive_sha256']
    config['base_output_dir']=f'/scratch/ql44/we2614/mts-spi-study/representation/synthetic/{RUN}/mpis'
    (ROOT/f'configs/external/{RUN}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    print('Saved',len(rows),'records',manifest['archive_sha256'])


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):build()
