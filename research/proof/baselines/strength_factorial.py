"""Cross physical/shared-input gain and independent sensor noise at M16/T1000."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mtsspi import generators

ROOT=Path(__file__).resolve().parents[3]
RUN='strength-factorial-261015'
DATA=ROOT/'data/proof'/RUN
OUT=ROOT/'results/proof'/RUN
PROTOCOL=ROOT/'research/proof/baselines/configs/analysis'/f'{RUN}.yaml'
REMOTE=f'/scratch/ql44/we2614/mts-spi-study/proof/{RUN}'
M,T,SEED=16,1000,2610152026
CLASSES=('VAR','Kuramoto','Wave','CML-intermittency','CML-defect','Gaussian-shared','Cauchy-shared')
ARMS=('fixed_clean','fixed_sensor','varying_clean','varying_sensor')


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def normalize(x):
    x=np.asarray(x,float)
    assert x.shape==(M,T) and np.isfinite(x).all()
    sd=x.std(1,keepdims=True)
    if np.any(sd<=1e-12):raise ValueError('Degenerate channel; preserve the failed seed')
    return (x-x.mean(1,keepdims=True))/sd


def nuisance(instance):
    gain=np.random.default_rng(np.random.SeedSequence([SEED,instance,81])).uniform(.5,1.5)
    attenuation=np.random.default_rng(np.random.SeedSequence([SEED,instance,82])).uniform(.25,.95)
    return float(gain),float(np.sqrt(1/attenuation-1)),float(attenuation)


def latent(label,instance,multiplier):
    index=CLASSES.index(label)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,instance,index,0]))
    common=dict(M=M,T=T,zscore=False,rng=rng)
    if label=='VAR':
        total=.2*multiplier
        assert .5+total<.98
        x=generators.generate_varma(**common,phi=.5,coupling=total/2,ma_phi=0,ma_coupling=0,noise_std=.1,transients=2000)
        parameter,value='total_neighbour',total
    elif label=='Kuramoto':
        value=3.5*multiplier
        x=generators.generate_kuramoto(**common,K=value,dt=.00625,omega_mean=3.,omega_std=1.73205,eta=0,
                                      connectivity='all-to-all',transients=2000,output='sin')
        parameter='K'
    elif label=='Wave':
        value=100*multiplier
        x=generators.generate_wave_1d(**common,c=np.sqrt(value),dt=.00125,n_modes=5,ic_decay=1.5,noise_std=0)
        parameter='c_squared'
    elif label.startswith('CML'):
        alpha,eps=(1.7522,.00115) if label=='CML-intermittency' else (1.895,.1)
        value=eps*multiplier
        x=generators.generate_cml_logistic(**common,alpha=alpha,eps=value,lattice_size=100,transients=2000)
        parameter='eps'
    else:
        draw=rng.normal if label=='Gaussian-shared' else rng.standard_cauchy
        innovations=draw(size=(M,T));common_input=draw(size=(1,T))
        loading=np.linspace(.5,1.5,M)[:,None]
        value=.75*multiplier
        x=(innovations+value*loading*common_input).T
        parameter='common_loading'
    return normalize(x.T),parameter,float(value)


def observations(label,instance):
    gain,eta,attenuation=nuisance(instance)
    noise=np.random.default_rng(np.random.SeedSequence([SEED,instance,CLASSES.index(label),1])).normal(size=(M,T))
    records=[]
    for multiplier,physical in [(1.,'fixed'),(gain,'varying')]:
        x,parameter,value=latent(label,instance,multiplier)
        for sigma,sensor in [(0.,'clean'),(eta,'sensor')]:
            y=normalize(x+sigma*noise)
            off=~np.eye(M,dtype=bool)
            records.append((y,dict(label=label,instance=instance,arm=physical+'_'+sensor,
                physical_multiplier=multiplier,parameter_name=parameter,parameter_value=value,
                sensor_sigma=sigma,nominal_attenuation=1. if sigma==0 else attenuation,
                mean_abs_r=float(abs(np.corrcoef(y)[off]).mean()),native_abs_r=float(abs(np.corrcoef(x)[off]).mean()),
                mean_lag1_autocorrelation=float(np.mean(np.sum(y[:,1:]*y[:,:-1],1)/(T-1))),
                interpretation='shared_input' if label.endswith('shared') else 'physical_coupling')))
    return records


def prepare():
    if (DATA/'manifest.json').exists():raise FileExistsError('Preserve immutable input bank')
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    arrays,rows={},[]
    for instance in range(40):
        for label in CLASSES:
            for x,settings in observations(label,instance):
                name=f'{settings["arm"]}-{label}-i{instance:02d}'
                arrays[name]=x
                rows.append(dict(row_id=name,corpus_index=len(rows),M=M,T=T,role='train' if instance<20 else 'evaluation',**settings))
        for index,kind in enumerate(['Gaussian','Cauchy']):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,instance,90+index]))
            x=normalize(rng.normal(size=(M,T)) if index==0 else rng.standard_cauchy(size=(M,T)))
            name=f'{kind}-independent-i{instance:02d}'
            arrays[name]=x
            rows.append(dict(row_id=name,corpus_index=len(rows),M=M,T=T,label=kind+'-independent',arm='reference',instance=instance,
                             role='reference',physical_multiplier=0.,parameter_name='none',parameter_value=0.,sensor_sigma=0.,nominal_attenuation=1.,
                             mean_abs_r=float(abs(np.corrcoef(x)[~np.eye(M,dtype=bool)]).mean()),interpretation='independent_reference'))
        print('Generated instance',instance,flush=True)
    np.savez_compressed(DATA/'observations.npz',**arrays,
        __dataset_names__=np.array(list(arrays)),__labels_json__=np.array([json.dumps([r['label']]) for r in rows]),
        __shapes__=np.array([[M,T]]*len(rows)),__axis_order__=np.array(['process','observation']))
    protocol=yaml.safe_load(PROTOCOL.read_text())
    manifest=dict(rows=rows,archive_sha256=sha(DATA/'observations.npz'),script_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),
                  generator_sha256={name:sha(ROOT/'src/mtsspi/generators'/name) for name in ['linear.py','dynamical.py','pde.py']},protocol=protocol)
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(OUT/'inputs.csv',index=False)
    config=dict(name=RUN,source=dict(format='named-npz-v1',archive=REMOTE+'/observations.npz',sha256=manifest['archive_sha256'],axis_order=['process','observation']),
                base_output_dir=REMOTE+'/mpis',pyspi_config='configs/pyspi/benchmarked_p90.yaml',normalise=False,random_seed=2610152027)
    (DATA/'corpus.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    # Select pilot by input noise level only, before any SPI outcomes.
    frame=pd.DataFrame(rows)
    pilot_instance=int(frame.loc[frame.sensor_sigma.idxmax(),'instance'])
    pilot=frame.index[(frame.instance==pilot_instance)&frame.arm.isin(['fixed_clean','varying_sensor','reference'])].to_numpy()+1
    (DATA/'pilot-indices.txt').write_text(''.join(f'{i}\n' for i in pilot))
    remaining=np.setdiff1d(np.arange(1,len(rows)+1),pilot)
    (DATA/'remaining-indices.txt').write_text(''.join(f'{i}\n' for i in remaining))
    receipt=dict(run=RUN,stage='prepared',records=len(rows),archive_sha256=manifest['archive_sha256'],protocol_sha256=manifest['protocol_sha256'],
                 source_sha256=manifest['script_sha256'],pilot_indices=pilot.tolist(),gadi_jobs=[])
    (OUT/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Prepared',len(rows),'records; no failed seeds; archive',manifest['archive_sha256'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare']);p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):prepare()
