"""Fresh, label-blind covariance-modulation construction and raw references."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from src.covariance_modulation import simulate, raw_references
from src.phase_surrogates import pooled_autospectrum
from src.representation_state_data import file_hash, observed_view


def build(config_path,output):
    if output.exists(): raise FileExistsError(output)
    cfg=yaml.safe_load(config_path.read_text())
    masters=[]; records=[]; rows=[]; corpus={}; features={}; targets=[]
    n=cfg['generator']['N_full']; count_per_cohort=max(cfg['sampling']['labelled_training_masters_per_coupling'])
    for split,role,count in [(0,'training_pool',cfg['sampling']['training_masters_per_coupling']),
                              (1,'evaluation',cfg['sampling']['evaluation_masters_per_coupling'])]:
        for fi,family in enumerate(cfg['generator']['families']):
            for k,(lo,hi) in enumerate(cfg['generator']['alpha_bins']):
                for rep in range(count):
                    seed=[cfg['sampling']['master_seed'],split,fi,k,rep]
                    rng=np.random.default_rng(seed+[1]); alpha=rng.uniform(lo,hi)
                    c=rng.uniform(.05,.15); base_d=rng.uniform(.25,.6); d=base_d*rng.uniform(.8,1.2,n)
                    x,s=simulate(alpha,c,d,family,seed+[2],t=1000)
                    order=np.random.default_rng(seed+[3]).permutation(n)
                    x=x[:,order].astype(np.float32); mid=f's{split}-{family}-k{k}-r{rep:02d}'
                    mi=len(masters); masters.append(x); targets.append(alpha)
                    cohort=rep//count_per_cohort if split==0 else None
                    records.append(dict(master_id=mid,role=role,family=family,seed_parts=seed,alpha=alpha,
                                        c=c,d=d.tolist(),sensor_order=order.tolist(),state_mean=float(s.mean()),
                                        coupling_index=k,replicate=rep,cohort_index=cohort))
                    cells=[cfg['observations']['source']] if split==0 else list(cfg['observations'].values())
                    for cell in cells:
                        if not isinstance(cell,dict): continue
                        m,t=cell['M'],cell['T']; view=observed_view(x,m,t); name=f'{mid}-M{m}-T{t}'
                        corpus[name]=view
                        rows.append(dict(row_id=name,master_id=mid,master_index=mi,role=role,family=family,
                                         coupling_index=k,cohort_index=cohort,M=m,T=t,target=alpha,corpus_index=len(rows)+1))
                        raw=raw_references(view);raw['spectrum']=pooled_autospectrum(view)
                        for key,value in raw.items(): features.setdefault(key,[]).append(value)
            print(f'{role}/{family}: {len(masters)} masters',flush=True)
    output.mkdir(parents=True)
    np.save(output/'masters.npy',np.stack(masters),allow_pickle=False)
    np.save(output/'targets.npy',np.asarray(targets),allow_pickle=False)
    np.save(output/'observables.npy',np.zeros((len(rows),2)),allow_pickle=False)
    np.savez_compressed(output/'raw-references.npz',**{k:np.asarray(v) for k,v in features.items()},
                        row_id=np.asarray([r['row_id'] for r in rows]))
    # Required by the existing generic statistical runner; unused legacy fields.
    np.savez_compressed(output/'raw-controls.npz',X_u=np.asarray(features['marginal']),memory=np.zeros(len(rows)),
                        references=np.zeros((len(rows),2)),row_id=np.asarray([r['row_id'] for r in rows]))
    corpus.update(__dataset_names__=np.asarray([r['row_id'] for r in rows]),__labels_json__=np.asarray(['[]']*len(rows)),
                  __shapes__=np.asarray([[r['T'],r['M']] for r in rows]),__axis_order__=np.asarray(['observation','process']))
    np.savez_compressed(output/'views.npz',**corpus)
    artifacts=['masters.npy','targets.npy','observables.npy','raw-references.npz','raw-controls.npz','views.npz']
    manifest=dict(config_sha256=file_hash(config_path),artifacts={p:file_hash(output/p) for p in artifacts},
                  rows=rows,masters=records,status='prospectively_specified_exploratory_pilot',
                  code_sha256={p:file_hash(Path(p)) for p in [__file__,'src/covariance_modulation.py']})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(masters=len(masters),views=len(rows),archive_sha256=manifest['artifacts']['views.npz'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.config,a.output)
