import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.empirical.ccep_pilot.cohort import build, native_qc
from research.empirical.ccep_pilot.prepare import clean_windows
from research.empirical.ccep_pilot.acquire import sha256


def test_short_baseline_keeps_original_guards():
    events=pd.DataFrame(columns=['onset','duration','trial_type'])
    assert clean_windows(events,49.999,32)==[]
    assert clean_windows(events,50,32)==[9.]
    events=pd.DataFrame([dict(onset=45.,duration=1.,trial_type='artifact')])
    assert clean_windows(events,60,32)==[]


def test_frozen_native_qc_rejects_shared_transient():
    rng=np.random.default_rng(261014);x=rng.normal(0,1e-5,(10,4096))
    assert native_qc(x)['qc_pass']
    x[:3,100:150]=.001
    assert not native_qc(x)['qc_pass']


def test_single_window_and_same_subject_different_runs(tmp_path):
    for folder in ('scope','prepared'): (tmp_path/folder).mkdir()
    reports=[];reviews={};rng=np.random.default_rng(5)
    for run in ('subject-run1','subject-run2'):
        x=rng.normal(size=(3,4096))
        path=tmp_path/'prepared'/f'{run}.npz';np.savez_compressed(path,w0=x,a0=x+np.arange(4096)/4096)
        reports.append(dict(run_id=run,subject='subject',channels=['A','B','C'],
            original_reference='G2',prepared_sha256=sha256(path),windows=[dict(window=0,start=9.,qc_pass=True)]))
        reviews[run]={'0':dict(accept=True,reason='Native/heatmap reviewed')}
    scope=tmp_path/'scope/runs.json';scope.write_text(json.dumps(reports))
    review=tmp_path/'review.json';review.write_text(json.dumps(dict(scope_sha256=sha256(scope),runs=reviews)))
    build(tmp_path,review);build(tmp_path,review)
    configs=json.loads((tmp_path/'active-configs.json').read_text());assert len(configs)==2
    for config in configs:
        with np.load(Path(config).parent/'observations.npz') as bank:
            names=bank['__dataset_names__'];assert len(names)==2
            assert np.allclose(bank[names[0]].std(axis=1),1)
            assert np.allclose(bank[names[0]].mean(axis=1),0)
