"""Independent-condition oscillatory records under an explicit generator protocol."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from src.oscillatory_coorganization import simulate,observed_controls
from src.covariance_modulation import raw_references
from src.phase_surrogates import pooled_autospectrum
from src.representation_state_data import observed_view,file_hash


def main(config,data):
    if data.exists():raise FileExistsError(data)
    cfg=yaml.safe_load(config.read_text());rows=[];records=[];masters=[];raw={};views={};observables=[]
    per_cohort=max(cfg['sampling']['labelled_training_masters_per_coupling'])
    for split,role,count in [(0,'training_pool',cfg['sampling']['training_masters_per_coupling']),
                            (1,'evaluation',cfg['sampling']['evaluation_masters_per_coupling'])]:
        for condition in [0,1]:
            for replicate in range(count):
                seed=[cfg['sampling']['master_seed'],split,condition,replicate]
                if cfg['generator']['name']=='oscillatory_direct_phase':
                    from src.oscillatory_mechanism import draw_parameters,simulate as simulate_direct
                    settings=cfg['generator']['settings'];meta=draw_parameters(seed+[0],settings)
                    x=simulate_direct(bool(condition),seed+[1],meta,settings,t=cfg['generator']['input_T'])
                else:
                    if cfg['generator']['name']!='oscillatory_coorganization':raise ValueError('unknown oscillator generator')
                    x,meta=simulate(bool(condition),seed,ranges=cfg['generator'].get('ranges'))
                mid=cfg.get('record_prefix','')+f's{split}-a{condition}-r{replicate:03d}'
                index=len(masters);masters.append(x);cohort=replicate//per_cohort if split==0 else None
                records.append(dict(master_id=mid,master_index=index,seed_parts=seed,role=role,
                                    target=condition,cohort_index=cohort,**meta))
                cells=[cfg['observations']['source']] if split==0 else list(cfg['observations'].values())
                for cell in cells:
                    m,t=cell['M'],cell['T'];view=observed_view(x,m,t);name=f'{mid}-M{m}-T{t}';views[name]=view
                    rows.append(dict(row_id=name,master_id=mid,master_index=index,role=role,family='oscillatory',
                                     coupling_index=condition,cohort_index=cohort,M=m,T=t,target=condition,corpus_index=len(rows)+1))
                    features={k:v for k,v in raw_references(view).items() if k!='moment_proxy'}
                    features['spectrum']=pooled_autospectrum(view);control=observed_controls(view)
                    features.update(phase=control['phase_summary'],envelope=control['envelope_summary'],agreement=control['direct_agreement'])
                    for key,value in features.items():raw.setdefault(key,[]).append(value)
                    observables.append(control['direct_agreement'])
    data.mkdir(parents=True);np.save(data/'masters.npy',np.stack(masters),allow_pickle=False)
    np.save(data/'observables.npy',np.asarray(observables),allow_pickle=False)
    np.savez_compressed(data/'raw-references.npz',**{k:np.asarray(v) for k,v in raw.items()},row_id=np.array([r['row_id'] for r in rows]))
    views.update(__dataset_names__=np.array([r['row_id'] for r in rows]),__labels_json__=np.array(['[]']*len(rows)),
                 __shapes__=np.array([[r['T'],r['M']] for r in rows]),__axis_order__=np.array(['observation','process']))
    np.savez_compressed(data/'views.npz',**views)
    manifest=dict(config_sha256=file_hash(config),rows=rows,masters=records,
        artifacts={p:file_hash(data/p) for p in ['masters.npy','observables.npy','raw-references.npz','views.npz']},
        code_sha256={p:file_hash(Path(p)) for p in [__file__,'src/oscillatory_coorganization.py']+
                     (['src/oscillatory_mechanism.py'] if cfg['generator']['name']=='oscillatory_direct_phase' else [])},
        status=cfg.get('status','fresh_pilot_frozen_before_generation_independent_records_no_pretraining'))
    (data/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dict(masters=len(masters),views=len(rows),sha256=manifest['artifacts']['views.npz']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    a=p.parse_args();main(a.config,a.data)
