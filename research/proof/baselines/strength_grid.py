"""Fresh size grid for the paired physical-coupling/sensor-noise experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from mtsspi import generators
from research.proof.baselines.strength_factorial import ROOT,sha,ARMS

RUN='strength-grid-261015'
DATA=ROOT/'data/proof'/RUN
OUT=ROOT/'results/proof'/RUN
REMOTE=f'/scratch/ql44/we2614/mts-spi-study/proof/{RUN}'
PROTOCOL=ROOT/'research/proof/baselines/configs/analysis'/f'{RUN}.yaml'
CLASSES=('VAR','Kuramoto','Wave','CML-intermittency','CML-frozen','Gaussian-shared','Cauchy-shared')
MS=(8,16,32);TS=(500,1000,2000);SEED=2610153026


def rng(m,t,instance,label,purpose):
    return np.random.default_rng(np.random.SeedSequence([SEED,m,t,instance,label,purpose]))


def normalize(x):
    assert np.isfinite(x).all()
    sd=x.std(1,keepdims=True)
    if np.any(sd<=1e-12):raise ValueError('Degenerate recording: preserve seed, do not replace')
    return (x-x.mean(1,keepdims=True))/sd


def latent(label,m,t,instance,gain):
    random=rng(m,t,instance,CLASSES.index(label),0)
    common=dict(M=32,T=t,zscore=False,rng=random)
    if label=='VAR':
        assert .5+.2*gain<.98
        x=generators.generate_varma(**common,phi=.5,coupling=.1*gain,ma_phi=0,ma_coupling=0,noise_std=.1,transients=2000)
        parameter,value='total_neighbour',.2*gain
    elif label=='Kuramoto':
        x=generators.generate_kuramoto(**common,K=3.5*gain,dt=.00625,omega_mean=3,omega_std=1.73205,eta=0,connectivity='all-to-all',transients=2000,output='sin')
        parameter,value='K',3.5*gain
    elif label=='Wave':
        x=generators.generate_wave_1d(**common,c=np.sqrt(100*gain),dt=.00125,n_modes=5,ic_decay=1.5,noise_std=0)
        parameter,value='c_squared',100*gain
    elif label.startswith('CML'):
        alpha,eps=(1.7522,.00115) if label=='CML-intermittency' else (1.45,.2)
        x=generators.generate_cml_logistic(**common,alpha=alpha,eps=eps*gain,lattice_size=100,transients=2000)
        parameter,value='eps',eps*gain
    else:
        draw=random.normal if label=='Gaussian-shared' else random.standard_cauchy
        innovations=draw(size=(32,t));driver=draw(size=(1,t))
        x=(innovations+.75*gain*np.linspace(.5,1.5,32)[:,None]*driver).T
        parameter,value='common_loading',.75*gain
    assert x.shape==(t,32) and np.isfinite(x).all()
    return x.T,parameter,float(value)


def observations(label,m,t,instance):
    gain=float(rng(m,t,instance,90,0).uniform(.5,1.5))
    attenuation=float(rng(m,t,instance,90,1).uniform(.25,.95))
    sigma=np.sqrt(1/attenuation-1)
    indices=rng(m,t,instance,90,2).permutation(32)[:m]
    noise=rng(m,t,instance,CLASSES.index(label),1).normal(size=(32,t))[indices]
    for multiplier,physical in [(1.,'fixed'),(gain,'varying')]:
        host,parameter,value=latent(label,m,t,instance,multiplier)
        x=normalize(host[indices]);mask=~np.eye(m,dtype=bool)
        for eta,sensor in [(0.,'clean'),(sigma,'sensor')]:
            y=normalize(x+eta*noise)
            yield y,dict(label=label,instance=instance,M=m,T=t,arm=physical+'_'+sensor,
                         parent=f'M{m}-T{t}-i{instance}',physical_multiplier=multiplier,
                         parameter_name=parameter,parameter_value=value,sensor_sigma=float(eta),
                         nominal_attenuation=1. if sensor=='clean' else attenuation,
                         mean_abs_r=float(abs(np.corrcoef(y)[mask]).mean()),
                         native_abs_r=float(abs(np.corrcoef(x)[mask]).mean()),observed_indices=indices.tolist())


def prepare():
    if (DATA/'manifest.json').exists():raise FileExistsError('Preserve immutable bank')
    DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    rows=[];cells=[]
    for m in MS:
        for t in TS:
            cell=f'M{m}-T{t}';cell_name=RUN+'-'+cell;arrays={};local=[]
            for instance in range(5):
                for label in CLASSES:
                    for x,settings in observations(label,m,t,instance):
                        name=f'{cell}-{settings["arm"]}-{label}-i{instance}'
                        arrays[name]=x
                        local.append(dict(row_id=name,cell=cell,cell_name=cell_name,local_index=len(local)+1,role='core',**settings))
                for index,kind in enumerate(['Gaussian','Cauchy']):
                    random=rng(m,t,instance,91+index,0)
                    x=normalize(random.normal(size=(m,t)) if index==0 else random.standard_cauchy(size=(m,t)))
                    name=f'{cell}-{kind}-independent-i{instance}';arrays[name]=x
                    local.append(dict(row_id=name,cell=cell,cell_name=cell_name,local_index=len(local)+1,role='reference',label=kind+'-independent',M=m,T=t,instance=instance,arm='reference',parent=f'{cell}-i{instance}',mean_abs_r=float(abs(np.corrcoef(x)[~np.eye(m,dtype=bool)]).mean())))
            path=DATA/f'{cell}.npz'
            np.savez_compressed(path,**arrays,__dataset_names__=np.array(list(arrays)),__labels_json__=np.array([json.dumps([r['label']]) for r in local]),__shapes__=np.array([[m,t]]*len(local)),__axis_order__=np.array(['process','observation']))
            digest=sha(path)
            config=dict(name=cell_name,source=dict(format='named-npz-v1',archive=f'{REMOTE}/{cell}.npz',sha256=digest,axis_order=['process','observation']),base_output_dir=REMOTE+'/mpis',pyspi_config='configs/pyspi/benchmarked_p90.yaml',normalise=False,random_seed=2610152027)
            (DATA/f'{cell}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
            # Input-defined resource/validity pilot: both extremes of intervention,
            # every class plus references, instance0. Never selected from SPI outcomes.
            selected=[r['local_index'] for r in local if r['instance']==0 and r['arm'] in ['fixed_clean','varying_sensor','reference']]
            (DATA/f'{cell}-pilot.txt').write_text(''.join(f'{i}\n' for i in selected))
            (DATA/f'{cell}-remaining.txt').write_text(''.join(f'{i}\n' for i in range(1,151) if i not in selected))
            for r in local:r['archive_sha256']=digest
            rows.extend(local);cells.append(dict(cell=cell,M=m,T=t,records=150,archive_sha256=digest,pilot_indices=selected))
            print('Prepared',cell,flush=True)
    manifest=dict(run=RUN,rows=rows,cells=cells,source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),generator_sha256={n:sha(ROOT/'src/mtsspi/generators'/n) for n in ['linear.py','dynamical.py','pde.py']})
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    pd.DataFrame(rows).to_csv(OUT/'inputs.csv',index=False)
    (OUT/'execution.json').write_text(json.dumps(dict(run=RUN,stage='prepared',records=len(rows),manifest_sha256=sha(DATA/'manifest.json'),source_sha256=sha(__file__),protocol_sha256=sha(PROTOCOL),cells=cells,jobs=[]),indent=2)+'\n')
    print('Prepared',len(rows),'records; no replaced seeds.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare']);parser.parse_args()
    prepare()
