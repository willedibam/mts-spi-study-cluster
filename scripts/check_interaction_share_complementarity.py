"""Retrospective fixed-weight prediction fusion; no selection on evaluation data."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.representation_screen import bootstrap_group_means
from src.representation_state_data import file_hash, load_state_data


def check(config, data, results, output):
    if output.exists():
        raise FileExistsError(output)
    protocol = yaml.safe_load(config.read_text())
    manifest, _ = load_state_data(data, config)
    rows = manifest['rows']
    expected_evaluation = np.asarray([i for i,row in enumerate(rows) if row['role']=='evaluation'])
    methods = {'linear': 'raw', 'nonlinear': 'raw',
               'random_encoder': 'random', 'z-pls': 'gadi-analysis/statistical'}
    seeds = protocol['methods']['subset_seeds']
    budgets = protocol['sampling']['labelled_training_masters_per_coupling']
    families = protocol['generator']['families']
    cells, primary, primary_boot, cohort_errors, provenance = {}, {}, {}, {}, []
    for family_index, family in enumerate(families):
        errors = {}
        for n in budgets:
            for seed in seeds:
                predictions = {}
                reference = None
                for method, directory in methods.items():
                    stem = results/directory/family/f'{method}-n{n}-s{seed}'
                    metadata = json.loads(stem.with_suffix('.json').read_text())
                    identity = metadata['identity']
                    assert identity['protocol_sha256'] == file_hash(config)
                    assert identity['manifest_sha256'] == file_hash(data/'manifest.json')
                    assert (identity['source_family'],identity['n_per_coupling'],identity['seed']) == (family,n,seed)
                    assert metadata['predictions_sha256'] == file_hash(stem.with_suffix('.npz'))
                    with np.load(stem.with_suffix('.npz')) as archive:
                        arrays = {key: archive[key] for key in ['train_indices','evaluation_indices','row_id','target','prediction']}
                    if reference is None:
                        reference = arrays
                    else:
                        for key in ['train_indices','evaluation_indices','row_id','target']:
                            np.testing.assert_array_equal(reference[key],arrays[key])
                    predictions[method] = arrays['prediction']
                    provenance.append({'path': str(stem.with_suffix('.json')), 'sha256': file_hash(stem.with_suffix('.json'))})
                train, indices = reference['train_indices'], reference['evaluation_indices']
                np.testing.assert_array_equal(indices,expected_evaluation)
                assert len(train) == n * len(protocol['generator']['nominal_shares'])
                assert all(rows[i]['role']=='training_pool' and rows[i]['family']==family for i in train)
                assert not set(train) & set(indices)
                assert not {rows[i]['master_id'] for i in train} & {rows[i]['master_id'] for i in indices}
                np.testing.assert_array_equal(reference['row_id'], [rows[i]['row_id'] for i in indices])
                np.testing.assert_array_equal(reference['target'], [rows[i]['target'] for i in indices])
                if protocol['sampling'].get('disjoint_training_cohorts'):
                    assert all(rows[i]['cohort_index']==seeds.index(seed) for i in train)
                for companion in ['z-pls','random_encoder','nonlinear']:
                    predictions['linear+'+companion] = (predictions['linear']+predictions[companion])/2
                for method, prediction in predictions.items():
                    assert np.isfinite(prediction).all() and np.all((prediction>=0)&(prediction<=1))
                    errors[method,n,seed] = abs(prediction-reference['target'])
        for destination in families:
            for observation in ['source','shift']:
                shape = protocol['observations'][observation]
                mask = np.asarray([rows[i]['family']==destination and (rows[i]['M'],rows[i]['T'])==(shape['M'],shape['T']) for i in indices])
                strata = np.asarray([rows[i]['coupling_index'] for i in indices[mask]])
                assert len(set(rows[i]['master_id'] for i in indices[mask])) == mask.sum()
                cell = f'{family}->{destination}/{observation}'
                cells[cell] = {}
                for method in predictions:
                    cohort = np.asarray([[errors[method,n,seed][mask] for seed in seeds] for n in budgets])
                    values = cohort.mean(axis=1)
                    boot = bootstrap_group_means(values,strata,2000,1729+family_index)
                    cells[cell][method] = dict(MAE=values.mean(axis=1).tolist())
                    if family!=destination and observation=='shift':
                        primary.setdefault(method,[]).append(values.mean(axis=1))
                        primary_boot.setdefault(method,[]).append(boot)
                        cohort_errors.setdefault(method,[]).append(cohort.mean(axis=2))
    primary = {method: np.mean(values,axis=0) for method,values in primary.items()}
    primary_boot = {method: np.mean(values,axis=0) for method,values in primary_boot.items()}
    comparisons = {}
    for other in ['linear','z-pls','linear+random_encoder','linear+nonlinear']:
        left = 'linear+z-pls'
        cohort_delta = np.concatenate(cohort_errors[left],axis=1)-np.concatenate(cohort_errors[other],axis=1)
        comparisons[left+'_minus_'+other] = dict(
            MAE_difference=(primary[left]-primary[other]).tolist(),
            conditional_95_CI=np.quantile(primary_boot[left]-primary_boot[other],[.025,.975],axis=-1).T.tolist(),
            cohort_wins_of_10=(cohort_delta<0).sum(axis=1).tolist(),
            cohort_differences=cohort_delta.tolist())
    report = dict(status='retrospective_fixed_half_half_fusion_not_fresh_confirmation',
        interpretation='No weights, partners or evaluation cells selected by score. Conditional pointwise intervals; no inaccessible-information claim.',
        total_label_budgets=protocol['sampling']['total_label_budgets'],
        primary={method:dict(MAE=values.tolist()) for method,values in primary.items()},
        comparisons=comparisons, cells=cells, provenance=provenance,
        protocol_sha256=file_hash(config), verifier_sha256=file_hash(Path(__file__)))
    output.mkdir(parents=True)
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = ['# Fixed-weight complementarity diagnostic','',
        'Retrospective: both datasets were previously evaluated. Every blend uses equal weights and the same labelled cohort; no additional labels or tuning. Lower MAE is better.','',
        '| Predictor | 10 labels | 20 labels | 40 labels |','|---|---:|---:|---:|']
    lines += ['| '+method+' | '+' | '.join(f'{x:.4f}' for x in values)+' |' for method,values in primary.items()]
    lines += ['','All source/family/observation cells, conditional paired intervals and cohort contrasts are in results.json. No future study is automatically triggered by this diagnostic.','']
    (output/'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['config','data','results','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();check(args.config,args.data,args.results,args.output)
