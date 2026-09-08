"""Matched origin-labelled phase controls; reuse intact MPIs, recompute surrogates."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from src.interaction_share_reference import fit_map, energy_share, own_memory
from src.phase_surrogates import phase_surrogate, pooled_autospectrum, spectrum_checks
from src.representation_screen import training_subsets
from src.representation_state_data import file_hash, load_state_data, observed_view, source_pool_for_seed


def build(config_path, output):
    if output.exists():
        raise FileExistsError(output)
    config=yaml.safe_load(config_path.read_text()); spec=config['phase_control']
    parent_root=Path(spec['origin_data']); parent_config=Path(spec['origin_config'])
    parent,masters=load_state_data(parent_root,parent_config)
    parent_protocol=yaml.safe_load(parent_config.read_text()); source_rows=parent['rows']
    strata=np.asarray([r['coupling_index'] for r in source_rows]); chosen=set()
    for family in config['generator']['families']:
        pool=np.asarray([i for i,r in enumerate(source_rows) if r['role']=='training_pool' and r['family']==family])
        for seed in config['methods']['subset_seeds']:
            cohort=source_pool_for_seed(source_rows,pool,parent_protocol,seed)
            chosen.update(training_subsets(strata,cohort,[4],seed)[4].tolist())
    chosen.update(i for i,r in enumerate(source_rows) if r['role']=='evaluation' and (r['M'],r['T'])==(16,1000))
    chosen=np.asarray(sorted(chosen)); assert len(chosen)==600
    rows=[]; records=[]; intact=[]
    for i in chosen:
        original=source_rows[i]; index=len(rows)
        assert (original['M'],original['T'])==(16,1000)
        rows.append({**original,'master_index':index,'corpus_index':index+1,'origin_row_index':int(i)})
        records.append({**parent['masters'][original['master_index']],
                        'origin_master_index':original['master_index'],'observed_input_only':True})
        intact.append(observed_view(masters[original['master_index']],16,1000))
    intact=np.asarray(intact); targets=np.asarray([r['target'] for r in rows])
    spectra=np.stack([pooled_autospectrum(x) for x in intact])
    output.mkdir(parents=True)
    with np.load(spec['origin_bank']) as archive:
        assert archive['manifest_sha256'].item()==file_hash(parent_root/'manifest.json')
        np.testing.assert_array_equal(archive['row_id'],[r['row_id'] for r in source_rows])
        intact_bank={key:archive[key][chosen] for key in ['X_m','X_g','X_z','X_validity']}
        intact_bank.update({key:archive[key] for key in ['spi_order','schema_sha256','feature_contract']})
    report={}; start=time.perf_counter()
    for arm_index,arm in enumerate(spec['arms']):
        root=output/arm;root.mkdir()
        arrays=[];memory=[];references=[];checks=[]
        for i,x in enumerate(intact):
            seed=[spec['seed'],records[i]['origin_master_index'],arm_index]
            y=x.copy() if arm=='intact' else phase_surrogate(x,seed,arm=='common_phase')
            check=spectrum_checks(x,y,arm!='independent_phase')
            np.testing.assert_allclose(pooled_autospectrum(y),spectra[i],atol=1e-12,rtol=1e-12)
            check['histogram_RMS_difference']=float(np.sqrt(np.mean((np.sort(x,axis=0)-np.sort(y,axis=0))**2)))
            check['mean_abs_correlation']=float(abs(np.corrcoef(y.T)[~np.eye(16,dtype=bool)]).mean())
            checks.append(check);arrays.append(y);memory.append(own_memory(y))
            estimates=[]
            for nonlinear in [False,True]:
                for ridge in config['methods']['raw_ridge_fractions']:
                    model=fit_map(y,nonlinear=nonlinear,ridge_fraction=ridge)
                    own,cross=model.energies(y)
                    estimates.append(energy_share(own,cross,16,32))
            references.append(estimates)
        np.save(root/'masters.npy',np.asarray(arrays),allow_pickle=False)
        np.save(root/'targets.npy',targets,allow_pickle=False)
        np.save(root/'observables.npy',np.asarray(memory)[:,None],allow_pickle=False)
        np.savez_compressed(root/'raw-controls.npz',X_u=spectra,memory=np.asarray(memory),
            references=np.asarray(references),ridge_fractions=config['methods']['raw_ridge_fractions'],
            row_id=np.asarray([r['row_id'] for r in rows]))
        corpus={r['row_id']:x for r,x in zip(rows,arrays,strict=True)}
        corpus.update(__dataset_names__=np.asarray([r['row_id'] for r in rows]),
            __labels_json__=np.asarray(['[]']*len(rows)),__shapes__=np.asarray([[1000,16]]*len(rows)),
            __axis_order__=np.asarray(['observation','process']))
        np.savez_compressed(root/'views.npz',**corpus)
        files=['masters.npy','targets.npy','observables.npy','raw-controls.npz','views.npz']
        manifest=dict(config_sha256=file_hash(config_path),artifacts={name:file_hash(root/name) for name in files},
            rows=rows,masters=records,arm=arm,status='origin_labels_not_surrogate_Jacobians',
            origin_manifest_sha256=file_hash(parent_root/'manifest.json'),
            code_sha256={name:file_hash(Path(name)) for name in [__file__,'src/phase_surrogates.py','src/interaction_share_reference.py']})
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if arm=='intact':
            np.savez_compressed(root/'features.npz',**intact_bank,row_id=np.asarray([r['row_id'] for r in rows]),
                manifest_sha256=file_hash(root/'manifest.json'))
            (root/'features.json').write_text(json.dumps(dict(artifact_sha256=file_hash(root/'features.npz'),
                origin_bank_sha256=file_hash(Path(spec['origin_bank'])),selected_rows=chosen.tolist(),
                status='exact_subset_of_intact_bank_no_recomputation'),indent=2)+'\n')
        report[arm]=dict(records=len(rows),checks=checks,manifest_sha256=file_hash(root/'manifest.json'))
        smoke=[next(r['corpus_index'] for r in rows if r['family']==family) for family in ['linear','tanh']]
        (root/'smoke-indices.txt').write_text(''.join(str(i)+'\n' for i in smoke))
        # Representative independent node gate: all bins/families, then deterministic fill.
        gate=[]
        for family in ['linear','tanh']:
            for k in range(5):
                gate += [r['corpus_index'] for r in rows if r['family']==family and r['coupling_index']==k][:4]
        gate += [r['corpus_index'] for r in rows if r['corpus_index'] not in gate][:8]
        (root/'node-indices.txt').write_text(''.join(str(i)+'\n' for i in sorted(gate)))
        print(f'{arm}: {len(rows)} records',flush=True)
    report['seconds']=time.perf_counter()-start
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['config','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();build(a.config,a.output)
