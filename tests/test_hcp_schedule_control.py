import numpy as np

from src.hcp_schedule_control import discordant_mask, fit_block_lookup, predict_block_lookup


def test_lookup_uses_source_labels_and_fixed_unknown_prior():
    fitted = fit_block_lookup([1, 1, 10, 10], [0, 0, 1, 1])
    np.testing.assert_array_equal(predict_block_lookup(fitted, [1, 10, 99]), [0, 1, .5])


def test_discordant_selection_is_common_and_requires_both_classes_per_person():
    blocks = [1, 10, 2, 13, 22, 1]
    y = [1, 0, 1, 1, 0, 1]
    people = ['a', 'a', 'a', 'b', 'b', 'c']
    np.testing.assert_array_equal(discordant_mask(blocks, y, people), [True, True, False, True, True, False])
