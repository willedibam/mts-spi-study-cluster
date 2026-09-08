"""Verify independent seeds, all nested views and representative exact replays."""
import json
from pathlib import Path
import numpy as np
import yaml
from src.oscillatory_coorganization import simulate
from src.representation_state_data import load_state_data,observed_view,source_pool_for_seed,file_hash
from src.representation_screen import training_subsets


def main():
    data=Path('data/oscillatory_coorganization_pilot_260909');config=Path('configs/analysis/oscillatory-coorganization-pilot-260909.yaml')
    cfg=yaml.safe_load(config.read_text());manifest,masters=load_state_data(data,config)
    rows=manifest['rows'];records=manifest['masters'];assert len(rows)==520 and len(records)==320
    assert len({tuple(r['seed_parts']) for r in records})==320
    with np.load(data/'views.npz',allow_pickle=False) as a:
        for r in rows:np.testing.assert_array_equal(a[r['row_id']],observed_view(masters[r['master_index']],r['M'],r['T']))
    for index in [0,60,120,220]:
        r=records[index];replay,_=simulate(bool(r['target']),r['seed_parts']);np.testing.assert_array_equal(masters[index],replay)
    y=np.array([r['target'] for r in rows]);pool=np.array([i for i,r in enumerate(rows) if r['role']=='training_pool']);cohorts=[]
    for seed in cfg['methods']['subset_seeds']:
        cohort=source_pool_for_seed(rows,pool,cfg,seed);assert len(cohort)==40
        for other in cohorts:assert not set(cohort)&other
        cohorts.append(set(cohort));previous=set()
        for n,train in training_subsets(y,cohort,[5,10,20],seed).items():
            assert len(train)==2*n and sum(y[train])==n and previous<=set(train);previous=set(train)
    result=dict(status='passed',exact_views=520,unique_independent_seeds=320,exact_generator_replays=4,
                disjoint_training_cohorts=3,nested_balanced_label_budgets=[10,20,40],manifest_sha256=file_hash(data/'manifest.json'))
    output=Path('results/oscillatory_coorganization_pilot_260909');output.mkdir(exist_ok=True)
    (output/'data-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
