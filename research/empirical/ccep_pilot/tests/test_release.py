import pytest
from research.empirical.ccep_pilot.release import resource_plan, pbs_measurement, EXPECTED_SOURCE, assert_no_active_farm


def test_actual_pbs_peak_releases_records_side_by_side():
    p=resource_plan(29160044/1024,12,5244,3,86)
    assert p['workers']==12 and p['groups']==3 and p['mem_gb']==105
    assert not p['fixed_worker_pilot'] and 'ncpus' not in p


def test_one_fixed_worker_pilot_then_no_escalation():
    p=resource_plan(12*1024,4,9000,2,64)
    assert p['workers']==8 and p['groups']==1 and p['fixed_worker_pilot']
    assert p['mem_gb']==25
    p=resource_plan(20*1024,8,4200,2,64)
    assert p['groups']==2 and p['workers']==8 and not p['fixed_worker_pilot']


def test_memory_reduces_simultaneous_records_not_workers():
    p=resource_plan(60*1024,12,100000,3,107)
    assert p['workers']==12 and p['groups']==2 and p['mem_gb']==150
    assert p['walltime']=='48:00:00'
    with pytest.raises(ValueError):resource_plan(160*1024,12,1000,3,107)
    assert resource_plan(30*1024,12,1000,0,107)['complete']


def test_pbs_measurement_requires_successful_single_matching_record(tmp_path):
    config=tmp_path/'external.yaml'
    job=dict(job_state='F',Exit_status=0,Variable_List=dict(CORPUS_CONFIG=str(config),
        PYSPI_SOURCE=EXPECTED_SOURCE,START_INDEX=2,END_INDEX=2,CORES_PER_RECORDING=12),
        resources_used=dict(mem='29160044kb',walltime='01:27:24'))
    peak,seconds=pbs_measurement(job,config,2,12)
    assert peak==pytest.approx(29160044/1024) and seconds==5244
    job['Variable_List']['END_INDEX']=3
    with pytest.raises(AssertionError):pbs_measurement(job,config,2,12)
    job['Variable_List']['END_INDEX']=2;job['job_state']='R'
    with pytest.raises(AssertionError):pbs_measurement(job,config,2,12)


def test_duplicate_active_farm_is_rejected(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    config=tmp_path/'external.yaml'
    result={'Jobs':{'123.gadi-pbs':dict(Job_Name='spi-farm',job_state='R',Variable_List={'CORPUS_CONFIG':str(config)})}}
    monkeypatch.setattr('subprocess.run',lambda *a,**k:SimpleNamespace(stdout=json.dumps(result)))
    with pytest.raises(RuntimeError,match='Existing farm'):assert_no_active_farm(config)
    result['Jobs']['123.gadi-pbs']['job_state']='F'
    assert_no_active_farm(config)
