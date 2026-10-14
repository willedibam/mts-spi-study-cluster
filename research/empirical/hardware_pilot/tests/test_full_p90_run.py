import pytest

from research.empirical.hardware_pilot.full_p90_run import resource_plan


def test_resource_plan_packs_without_unused_cores_or_memory_overflow():
    plan=resource_plan(10*1024,1200,44)
    assert plan['workers']==4 and plan['groups']==12 and plan['mem_gb']==156
    assert plan['walltime']=='00:40:00'
    tail=resource_plan(10*1024,1200,8)
    assert tail['groups']==8 and tail['mem_gb']==104
    memory_limited=resource_plan(20*1024,1200,44)
    assert memory_limited['groups']==7 and memory_limited['mem_gb']==175


@pytest.mark.parametrize('peak,seconds',[(0,1200),(1024,0),(float('nan'),1200),(200*1024,1200)])
def test_resource_plan_rejects_invalid_or_oversized_measurements(peak,seconds):
    with pytest.raises(ValueError):resource_plan(peak,seconds,44)
