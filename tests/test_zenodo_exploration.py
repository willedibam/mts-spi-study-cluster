"""Checks for the two details that can silently change the historical comparison."""
import numpy as np

from scripts.explore_zenodo_geometry import historical_input, neighbors, overlap
from scripts.refine_zenodo_clusters import best_jaccard
from scripts.render_zenodo_shortlist import exact_prefix_pairs


def test_historical_filter_order_and_raw_zero_imputation():
    x = np.arange(60, dtype=float).reshape(10, 6)
    x[-1, 1:] = np.nan  # Less than 20% observed: removed before column filtering.
    x[:2, 1] = np.nan  # 7/9 observations: fails the 80% column threshold.
    x[0, 2] = np.nan  # 8/9 observations: retained and imputed with raw zero.
    x = np.column_stack([x, x[:, 3]])  # Duplicate feature, not duplicate dataset.
    filled, rows, columns = historical_input(x)
    np.testing.assert_array_equal(rows, np.arange(9))
    np.testing.assert_array_equal(columns, [0, 2, 3, 4, 5])
    assert filled.shape == (9, 5)
    assert filled[0, 1] == 0
    assert filled[-1, -1] == 53  # No normalization was introduced.


def test_neighbours_remove_self_by_index_even_for_duplicate_mts():
    x = np.array([[0., 0.], [0., 0.], [1., 0.], [3., 0.]])
    nn = neighbors(x, k=2)
    assert all(i not in row for i, row in enumerate(nn))
    assert nn.shape == (4, 2)
    assert 1 in nn[0] and 0 in nn[1]
    assert overlap(nn, nn) == 1


def test_membership_stability_ignores_labels_but_penalizes_split_merge():
    members = np.array([0, 1, 2, 3])
    assert best_jaccard(members, np.array([8, 8, 8, 8, 2])) == 1
    assert best_jaccard(members, np.array([8, 8, 4, 4, 2])) == .5
    assert best_jaccard(members, np.array([8, 8, 8, 8, 8])) == .8
    assert best_jaccard(members, np.full(5, -1)) == 0


def test_window_overlap_requires_exact_values_and_same_channels():
    full = np.arange(30).reshape(3, 10)
    altered = full[:, :5].copy()
    altered[0, 0] += 1
    pairs = exact_prefix_pairs({"full":full, "prefix":full[:, :5],
                                "different":altered, "other_channels":full[:2, :5]})
    assert pairs == [{"short":"prefix", "long":"full", "shared_samples":5}]
