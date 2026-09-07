"""Audit observed views, future labels and training cohorts before extraction."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.check_interaction_share_feasibility import interaction_share
from scripts.check_interaction_share_references import parameters, simulate
from src.representation_screen import training_subsets
from src.representation_state_data import file_hash, load_state_data, observed_view, source_pool_for_seed


def check(config, data, output):
    protocol = yaml.safe_load(config.read_text())
    manifest, masters = load_state_data(data, config)
    records, rows = manifest['masters'], manifest['rows']
    targets = np.load(data/'targets.npy')
    with np.load(data/'views.npz') as views:
        for row in rows:
            np.testing.assert_array_equal(views[row['row_id']],
                observed_view(masters[row['master_index']], row['M'], row['T']))
            assert row['target'] == targets[row['master_index']]
    replayed = []
    for family in protocol['generator']['families']:
        for role in ['training_pool', 'evaluation']:
            i = next(i for i,r in enumerate(records) if r['family']==family and r['role']==role)
            r = records[i]
            a,b,gain = parameters(r['nominal_share'], 'independent_gain', r['seed_parts']+[2])
            np.testing.assert_array_equal([a,b,gain], [r['a'],r['b'],r['gain']])
            states,jac = simulate(family,a,b,r['seed_parts']+[0])
            np.testing.assert_array_equal(states[:1000,r['sensor_order']],masters[i])
            assert interaction_share(jac[1000:]) == targets[i]
            replayed.append(r['master_id'])
    bins = protocol['generator'].get('nominal_share_bins')
    if bins:
        for i,r in enumerate(records):
            low,high = bins[r['coupling_index']]
            assert low <= r['nominal_share'] < high
            if r['family']=='linear':
                np.testing.assert_allclose(targets[i], r['nominal_share'], atol=1e-12)
    strata = np.asarray([r['coupling_index'] for r in rows])
    for family in protocol['generator']['families']:
        pool = np.asarray([i for i,r in enumerate(rows) if r['family']==family and r['role']=='training_pool'])
        used = set()
        for seed in protocol['methods']['subset_seeds']:
            cohort = source_pool_for_seed(rows,pool,protocol,seed)
            subsets = training_subsets(strata,cohort,protocol['sampling']['labelled_training_masters_per_coupling'],seed)
            last = set()
            for n,subset in sorted(subsets.items()):
                assert last <= set(subset) and len(subset)==n*len(protocol['generator']['nominal_shares'])
                last = set(subset)
            if protocol['sampling'].get('disjoint_training_cohorts'):
                assert used.isdisjoint(last)
            used.update(last)
    for i,r in enumerate(records):
        if r['role']=='evaluation':
            pair = [row for row in rows if row['master_index']==i]
            assert len(pair)==2
            source,shift = [observed_view(masters[i],row['M'],row['T']) for row in pair]
            np.testing.assert_array_equal(source[-shift.shape[0]:,:shift.shape[1]],shift)
    result = dict(status='passed', masters=len(records), views=len(rows),
        exact_future_label_replays=replayed, disjoint_cohorts=protocol['sampling'].get('disjoint_training_cohorts',False),
        manifest_sha256=file_hash(data/'manifest.json'), protocol_sha256=file_hash(config),
        verifier_sha256=file_hash(Path(__file__)))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['config','data','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();check(args.config,args.data,args.output)
