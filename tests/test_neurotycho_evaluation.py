import numpy as np
import pytest
from src.neurotycho_evaluation import summarize_run, verify_pearson_edges


def test_equal_date_then_animal_weight_and_fixed_threshold():
    # One long failed date must not outweigh its animal's shorter successful date.
    y = np.tile([0, 1], 7)
    dates = np.array(['a1']*8 + ['a2']*2 + ['b1']*4)
    animals = np.array(['a']*10 + ['b']*4)
    p = y.astype(float); p[:8] = 1-y[:8]
    result = summarize_run(y, p, animals, dates, 16, 2000)
    assert result['mean']['balanced_accuracy'] == .75
    assert result['mean']['balanced_brier'] == .25
    assert len(result['dates']) == 3 and len(result['animals']) == 2


def test_edge_alignment_detects_corrupted_pair_feature():
    x = np.random.default_rng(9).normal(size=(6, 4))
    x = (x-x.mean(0))/x.std(0)
    z = np.corrcoef(x.T)[np.triu_indices(4, 1)].astype(np.float32)
    bank = dict(edges=x[None].astype(np.float32), z=z[None], validity=np.ones((1, 4), dtype=bool),
                lengths=np.array([6]), M=np.array([3]))
    assert verify_pearson_edges(bank) < 2e-6
    bank['z'][0, 0] += .01
    with pytest.raises(AssertionError):
        verify_pearson_edges(bank)
