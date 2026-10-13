import pytest
from research.empirical.ccep_pilot.release import resource_plan, recording_peak_mib


def test_measured_peaks_cover_every_simultaneous_record():
    p=resource_plan(12*1024,4,2100,3)
    assert p==dict(workers=4,groups=3,ncpus=48,mem_gb=45,walltime='01:10:00',remeasure=False,complete=False)


def test_worker_increase_is_one_remaining_record_for_remeasurement():
    p=resource_plan(21*1024,4,1800,3)
    assert p['workers']==6 and p['groups']==1 and p['remeasure']
    assert p['mem_gb']==180


def test_node_imbalance_and_limits_are_explicit():
    p=resource_plan(60*1024,24,100000,3)
    assert p['ncpus']==96 and p['mem_gb']==300 and p['walltime']=='48:00:00'
    with pytest.raises(ValueError):resource_plan(191*1024,4,2000,3)
    assert resource_plan(3*1024,1,2000,0)['complete']


def test_actual_runner_metadata_and_changed_worker_memory():
    peak={'parent':702.1,'largest_worker':12641.8}
    assert recording_peak_mib(peak,4)==pytest.approx(51269.3)
    p=resource_plan(peak,4,20506,3)
    assert p['workers']==12 and p['remeasure'] and p['memory_limited_workers']
    assert (peak['parent']+12*peak['largest_worker'])/1024*1.25<190
    # At that measured footprint, one simultaneous recording per node is necessary.
    p=resource_plan(peak,12,7200,2)
    assert not p['remeasure'] and p['groups']==2 and p['ncpus']==96
    assert p['mem_gb']>=recording_peak_mib(peak,12)/1024*2*1.25


def test_worker_count_reduces_when_memory_would_exceed_node():
    p=resource_plan({'parent':700.,'largest_worker':9000.},48,1800,2)
    assert p['workers']==16 and p['groups']==1 and p['remeasure']
