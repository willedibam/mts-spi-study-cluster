import pytest
from research.empirical.ccep_pilot.release import resource_plan


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
