import numpy as np

from scripts.diagnose_oscillatory_transfer_scores import balanced_batch_assignments, expected_balanced_accuracy


def test_ties_do_not_exploit_class_order():
    assignments, cutoff = balanced_batch_assignments(np.zeros(200))
    np.testing.assert_array_equal(assignments, np.full(200, .5))
    truth = np.repeat([0, 1], 100)
    assert expected_balanced_accuracy(truth, assignments) == .5
    assert cutoff == 0


def test_cutoff_ties_and_permutation_equivariance():
    scores = np.array([0., 1., 1., 1., 1., 2.])
    expected = np.array([0., .5, .5, .5, .5, 1.])
    assignments, _ = balanced_batch_assignments(scores)
    np.testing.assert_array_equal(assignments, expected)
    order = np.array([5, 2, 0, 4, 1, 3])
    shuffled, _ = balanced_batch_assignments(scores[order])
    np.testing.assert_array_equal(shuffled, expected[order])
    monotone, _ = balanced_batch_assignments(np.exp(scores))
    np.testing.assert_array_equal(monotone, expected)
