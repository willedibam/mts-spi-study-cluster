"""The interpretable Kuramoto family must not select on display observations."""
import numpy as np
import matplotlib.pyplot as plt
from mtsspi.figures import raincloud
from research.proof.p90.proof_composite import select_examples


def test_focused_kuramoto_selection_remaps_columns_and_freezes_orientation(tmp_path):
    labels = np.repeat(['kuramoto_omega-fast', 'kuramoto_omega-slow'], 10)
    x = np.zeros((20, 3))
    x[10:16, 0] = 1  # Perfect but outside the named temporal candidate family.
    x[10:16, 2] = 1  # Second eligible coordinate, original bank column 2.
    x[6:10, 2] = 1   # Reverse the selected feature on display observations.
    bank = {'instance':np.tile(np.arange(10), 2), 'y':labels, 'X_sym':x,
            'pairs':np.array([['cov_A', 'cov_B'],
                              ['corr_pearson_tau-1', 'corr_pearson_tau-2'],
                              ['corr_pearson_tau-10', 'corr_pearson_tau-20']])}
    pair = ('kuramoto_omega-fast', 'kuramoto_omega-slow')
    selected = select_examples(bank, tmp_path, pairs=[pair]*3)
    assert selected.feature_index.tolist() == [2, 2, 2]
    assert selected.spi_a.tolist() == ['corr_pearson_tau-10']*3
    assert selected.training_auc_oriented.tolist() == [1, 1, 1]
    assert selected.display_auc_fixed_orientation.tolist() == [0, 0, 0]


def test_fixed_coordinate_is_not_reselected(tmp_path):
    labels = np.repeat(['a', 'b'], 10)
    x = np.zeros((20, 2))
    x[10:, 0] = 1  # A stronger but unrequested coordinate.
    bank = {'instance':np.tile(np.arange(10), 2), 'y':labels, 'X_sym':x,
            'pairs':np.array([['x', 'y'], ['x', 'z']])}
    selected = select_examples(bank, tmp_path, pairs=[('a', 'b')]*3,
                               feature_pairs=[('x', 'z')]*3)
    assert selected.feature_index.tolist() == [1]*3
    assert selected.training_auc_oriented.tolist() == [.5]*3


def test_raincloud_keeps_sizes_aligned_when_values_are_missing():
    fig, ax = plt.subplots()
    raincloud(ax, [1, np.nan, 3], 0, 'black', np.random.default_rng(0),
              point_sizes=[8, 16, 32])
    points = ax.collections[0]
    np.testing.assert_array_equal(points.get_offsets()[:, 1], [1, 3])
    np.testing.assert_array_equal(points.get_sizes(), [8, 32])
    plt.close(fig)
