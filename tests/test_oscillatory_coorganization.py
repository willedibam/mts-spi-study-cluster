import numpy as np
from src.oscillatory_coorganization import partitions,simulate,observed_controls


def test_only_cross_partition_correspondence_changes():
    graphs=[]
    for aligned in [False,True]:
        p,a=partitions(aligned);same_p=p[:,None]==p[None,:];same_a=a[:,None]==a[None,:]
        assert np.array_equal(same_p.sum(1),np.full(32,8))
        assert np.array_equal(same_a.sum(1),np.full(32,8))
        mask=~np.eye(32,dtype=bool)
        np.testing.assert_array_equal(np.sort(same_p[mask]),np.sort(same_a[mask]))
        graphs.append(np.corrcoef(same_p[mask],same_a[mask])[0,1])
    np.testing.assert_allclose(graphs,[-3/28,1],atol=1e-14)


def test_paired_conditions_share_nuisance_and_sensor_order():
    x,a=simulate(False,123);y,b=simulate(True,123)
    assert {k:v for k,v in a.items() if k!='envelope_group'}=={k:v for k,v in b.items() if k!='envelope_group'}
    assert not np.array_equal(x,y)
    replay,_=simulate(False,123);np.testing.assert_array_equal(x,replay)


def test_observed_agreement_invariant_to_sensor_relabeling():
    x,_=simulate(True,123);x=x[:500,:8];a=observed_controls(x)
    b=observed_controls(x[:,[4,2,7,0,1,6,5,3]])
    for key in ['phase_summary','envelope_summary','direct_agreement']:
        np.testing.assert_allclose(a[key],b[key],atol=1e-12)
