import numpy as np
import pandas as pd
import pytest
from scripts.cml2d_full_observation import coverage_gate
from scripts.launch_cml2d_full_production import resource_plan


def test_gate_uses_seed_coverage_and_distinguishes_transfer():
    frame=pd.DataFrame([dict(r=r,role=role,eligible=True) for r in range(17)
        for role,n in [('development',8),('evaluation',32)] for _ in range(n)])
    assert coverage_gate(frame,{'passes_one_coordinate_gate':True})['passes']
    assert not coverage_gate(frame,{'passes_one_coordinate_gate':False})['passes']
    assert coverage_gate(frame,{'passes_one_coordinate_gate':False},False)['passes']
    indices=frame.index[(frame.r==0)&(frame.role=='evaluation')][:9]
    frame.loc[indices,'eligible']=False
    assert not coverage_gate(frame,{'passes_one_coordinate_gate':True})['passes']


def test_farm_plan_is_bounded_and_respects_memory():
    for timings in [[1000.]*24, [1000.]*23+[2100.]]:
        plan=resource_plan(timings)
        assert plan['ncpus']%48==0 and plan['memory_gb']>=8*plan['workers']
        assert plan['workers'] in (336,680)
    with pytest.raises(ValueError):resource_plan([1.]*23)
    with pytest.raises(ValueError):resource_plan([30000.]*24)
