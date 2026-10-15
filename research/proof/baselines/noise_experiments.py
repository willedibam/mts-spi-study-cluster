"""Paired eta/heterogeneity sweep and stronger random attenuation, fixed beforehand."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from mtsspi import generators
from research.proof.baselines.strength_factorial import ROOT,sha,normalize

RUN='noise-experiments-261016';DATA=ROOT/'data/proof'/RUN;OUT=ROOT/'results/proof'/RUN
PROTOCOL=ROOT/'research/proof/baselines/configs/analysis'/f'{RUN}.yaml'
REMOTE=f'/scratch/ql44/we2614/mts-spi-study/proof/{RUN}'
GDATA=f'/g/data/ql44/we2614/mts-spi-study/proof/{RUN}'
PYSPI='/scratch/ql44/we2614/mts-spi-study/operations/sources/pyspi-strength-factorial-c2d723f'
CLASSES=('VAR','Kuramoto','Wave','CML-intermittency','CML-frozen','Gaussian-shared','Cauchy-shared')
ETAS=(0,.5,1,2,4,8);M,T,N,SEED=16,1000,32,2610162026

def rng(i,k,p):return np.random.default_rng(np.random.SeedSequence([SEED,i,k,p]))

def latent(label,i):
    random=rng(i,CLASSES.index(label),0);common=dict(M=M,T=T,zscore=False,rng=random)
    if label=='VAR':x=generators.generate_varma(**common,phi=.5,coupling=.1,ma_phi=0,ma_coupling=0,noise_std=.1,transients=2000)
    elif label=='Kuramoto':x=generators.generate_kuramoto(**common,K=3.5,dt=.00625,omega_mean=3.,omega_std=1.73205,eta=0,connectivity='all-to-all',transients=2000,output='sin')
    elif label=='Wave':x=generators.generate_wave_1d(**common,c=10,dt=.00125,n_modes=5,ic_decay=1.5,noise_std=0)
    elif label.startswith('CML'):
        alpha,eps=(1.7522,.00115) if label=='CML-intermittency' else (1.45,.2)
        x=generators.generate_cml_logistic(**common,alpha=alpha,eps=eps,lattice_size=100,transients=2000)
    else:
        draw=random.normal if label=='Gaussian-shared' else random.standard_cauchy
        x=(draw(size=(M,T))+.75*np.linspace(.5,1.5,M)[:,None]*draw(size=(1,T))).T
    return normalize(x.T)

def observations(label,i):
    x=latent(label,i);e=rng(i,CLASSES.index(label),1).normal(size=(M,T))
    w=rng(i,90,2).exponential(size=M);w/=np.sqrt(np.mean(w*w))
    u=rng(i,90,3).uniform();off=~np.eye(M,dtype=bool)
    settings=[('dose','clean','equal',0.,1.)]
    for eta in ETAS[1:]:
        for profile in ['equal','exponential']:
            settings.append(('dose',f'{profile}-eta{eta:g}',profile,float(eta),1/(1+eta*eta)))
    settings.append(('dose','noise-limit','limit',None,0.))
    for arm,alpha in [('random-previous',.25+.70*u),('random-stronger',.05+.45*u)]:
        settings.append(('random',arm,'equal',float(np.sqrt(1/alpha-1)),float(alpha)))
    for experiment,arm,profile,eta,alpha in settings:
        sd=np.zeros(M) if eta is None else eta*(w if profile=='exponential' else np.ones(M))
        noise=e if eta is None else sd[:,None]*e
        y=normalize(e if eta is None else x+noise)
        expected=None if eta is None else float(np.mean(sd*sd))
        if eta is not None:assert np.isclose(expected,eta**2,rtol=1e-12,atol=1e-12)
        yield y,dict(experiment=experiment,arm=arm,profile=profile,eta=eta,nominal_equal_attenuation=alpha,
                     channel_noise_sd=sd.tolist() if eta is not None else None,expected_noise_power=expected,
                     realized_noise_power=float(np.mean(noise*noise)),mean_abs_r=float(abs(np.corrcoef(y)[off]).mean()),
                     latent_abs_r=float(abs(np.corrcoef(x)[off]).mean()),instance=i,label=label,parent=f'{label}-i{i:02d}',M=M,T=T)

def prepare():
    if (DATA/'manifest.json').exists():raise FileExistsError('Preserve input bank')
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    banks={k:{} for k in ['dose','random']};rows=[]
    for i in range(N):
        for label in CLASSES:
            for y,settings in observations(label,i):
                k=settings['experiment'];name=f'{settings["arm"]}-{label}-i{i:02d}';banks[k][name]=y
                rows.append(dict(row_id=name,local_index=len(banks[k]),corpus_name=RUN+'-'+k,role='train' if i<16 else 'evaluation',mpi_root=(GDATA if k=='dose' else REMOTE)+'/mpis',**settings))
        if i%8==0:print('Generated',i,flush=True)
    for k,arrays in banks.items():
        np.savez_compressed(DATA/(k+'.npz'),**arrays,__dataset_names__=np.array(list(arrays)),__labels_json__=np.array([json.dumps([r['label']]) for r in rows if r['experiment']==k]),__shapes__=np.array([[M,T]]*len(arrays)),__axis_order__=np.array(['process','observation']))
        digest=sha(DATA/(k+'.npz'))
        for r in rows:
            if r['experiment']==k:r['archive_sha256']=digest
        config=dict(name=RUN+'-'+k,source=dict(format='named-npz-v1',archive=REMOTE+'/'+k+'.npz',sha256=digest,axis_order=['process','observation']),base_output_dir=(GDATA if k=='dose' else REMOTE)+'/mpis',pyspi_config='configs/pyspi/benchmarked_p90.yaml',normalise=False,random_seed=2610152027)
        (DATA/(k+'.yaml')).write_text(yaml.safe_dump(config,sort_keys=False))
        pilot=[r['local_index'] for r in rows if r['experiment']==k and r['instance']==0 and (k=='random' or r['arm'] in ['clean','equal-eta8','exponential-eta8','noise-limit'])]
        (DATA/(k+'-pilot.txt')).write_text(''.join(f'{j}\n' for j in pilot))
        (DATA/(k+'-remaining.txt')).write_text(''.join(f'{j}\n' for j in range(1,len(arrays)+1) if j not in pilot))
    assert len(rows)==3136 and len(banks['dose'])==2688 and len(banks['random'])==448
    manifest=dict(run=RUN,rows=rows,protocol_sha256=sha(PROTOCOL),source_sha256=sha(__file__),generator_sha256={name:sha(ROOT/'src/mtsspi/generators'/name) for name in ['linear.py','dynamical.py','pde.py']})
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(rows).to_csv(OUT/'inputs.csv',index=False)
    receipt=dict(run=RUN,stage='prepared',records=len(rows),manifest_sha256=sha(DATA/'manifest.json'),protocol_sha256=sha(PROTOCOL),source_sha256=sha(__file__),pyspi_source=PYSPI,jobs=[])
    (OUT/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Prepared',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare']);p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):prepare()
