"""Check matched-support search against enumeration and participant weighting."""
import itertools
import numpy as np
import pandas as pd
from research.empirical.ccep_pilot.cohort_interpret import witness, participant_medians


def test_support_search_matches_brute_force():
    rng=np.random.default_rng(19)
    for _ in range(12):
        vectors=rng.integers(0,4,size=(10,6));subjects=np.repeat(np.arange(5),2)
        valid=[c for c in itertools.combinations(range(10),4) if len(set(subjects[list(c)]))==4]
        maximum=max(np.min(vectors[list(c)],axis=0).sum() for c in valid)
        for required in range(1,10):
            found=witness(vectors,subjects,required)
            assert (found is not None)==(maximum>=required)
            if found is not None:
                assert len(set(subjects[found]))==4
                assert np.min(vectors[found],axis=0).sum()>=required


def test_repeated_runs_do_not_supply_independent_participants():
    vectors=np.full((6,2),8)
    assert witness(vectors,['a','a','b','b','c','c'],8) is None
    assert witness(vectors,['a','a','b','b','c','d'],8) is not None


def test_window_count_does_not_weight_run_summary():
    frame=pd.DataFrame(dict(subject=['a']*4+['b'],run=['r1']*3+['r2','r3'],z=[0,0,0,1,.2]))
    result=participant_medians(frame,['z'])
    assert result.loc['a','z']==.5
    assert result.loc['b','z']==.2
