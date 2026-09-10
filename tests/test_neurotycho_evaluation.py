import numpy as np
from src.neurotycho_evaluation import summarize_run


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
