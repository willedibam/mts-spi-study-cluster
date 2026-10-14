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


def test_cohort_plan_exact_cores_and_pbs_memory():
    from research.empirical.ccep_pilot.cohort_dispatch import plan
    p=plan(41.33*1024,8100,107,4)
    assert p==dict(workers=12,groups=3,mem_gb=156,walltime='04:30:00')
    assert plan(12*1024,2700,48,1)==dict(workers=8,groups=1,mem_gb=15,walltime='01:30:00')


def test_dispatch_does_not_retry_ambiguous_receipt(tmp_path,monkeypatch):
    import research.empirical.ccep_pilot.cohort_dispatch as cd
    folder=tmp_path/'dispatch'/'subject-m48';folder.mkdir(parents=True)
    receipt=folder/'measurement.json';receipt.write_text('{"status":"requires-inspection"}')
    monkeypatch.setenv('PYSPI_SOURCE',cd.EXPECTED_SOURCE)
    monkeypatch.setattr(cd,'cohorts',lambda root:{('subject',48):[(None,None,None)]})
    monkeypatch.setattr(cd,'assert_no_active_farm',lambda config: (_ for _ in ()).throw(AssertionError('must not reach submission')))
    cd.dispatch(tmp_path)
    assert json.loads(receipt.read_text())['status']=='requires-inspection'


def test_prepare_accepts_pilot_header_rounding(tmp_path,monkeypatch):
    import research.empirical.ccep_pilot.cohort as cc
    raw_root=tmp_path/'input';raw=(raw_root/'raw');raw.mkdir(parents=True)
    prefix='sub-test_run-1';fs=2048;duration=50
    t=np.arange(fs*duration)/fs
    x=np.stack([20*np.sin(2*np.pi*(j+4)*t)+np.cos(2*np.pi*t) for j in range(3)],axis=1)
    (raw/f'{prefix}_ieeg.eeg').write_bytes(x.astype('<f4').tobytes())
    (raw/f'{prefix}_ieeg.vhdr').write_text('Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nDataFile='+prefix+'_ieeg.eeg\nMarkerFile='+prefix+'_ieeg.vmrk\nDataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=3\nSamplingInterval=4.882812e+02\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\nCh1=A,,1,µV\nCh2=B,,1,µV\nCh3=C,,1,µV\n')
    (raw/f'{prefix}_ieeg.vmrk').write_text('Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nDataFile='+prefix+'_ieeg.eeg\n[Marker Infos]\n')
    (raw/f'{prefix}_ieeg.json').write_text(json.dumps(dict(SamplingFrequency=fs,iEEGReference='G2')))
    pd.DataFrame(dict(name=['A','B','C'],type=['ECOG']*3,status=['good']*3)).to_csv(raw/f'{prefix}_channels.tsv',sep='\t',index=False)
    pd.DataFrame([dict(onset=50.,duration=0.,trial_type='electrical_stimulation',sample_start=102400)]).to_csv(raw/f'{prefix}_events.tsv',sep='\t',index=False)
    files=[dict(path=p.name,sha256=sha256(p)) for p in raw.iterdir()]
    selection=tmp_path/'selection.json';selection.write_text(json.dumps(dict(files=files,records=[dict(run_id=prefix,subject='sub-test',path=prefix+'_ieeg.json',duration=duration,starts=[9.])])) )
    (raw_root/'acquisition.json').write_text(json.dumps(dict(selection_sha256=sha256(selection),files=files)))
    monkeypatch.setattr(cc,'CONFIG',selection);monkeypatch.setattr(cc,'gallery',lambda *args:None)
    output=tmp_path/'output';cc.prepare(output,raw_root)
    report=json.loads((output/'scope/runs.json').read_text())[0]
    assert report['M']==3 and report['windows'][0]['qc_pass']
    with np.load(output/'prepared'/f'{prefix}.npz') as bank:
        assert bank['w0'].shape==(3,4096)
