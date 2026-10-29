"""Proof figure: fixed contrasts, size/shape encodings and frozen configuration."""
import numpy as np
import matplotlib.pyplot as plt
import pytest
from mtsspi.figures import raincloud
from research.proof.p90.proof_figure import T_MARKERS, load_config, marker_area, select_contrasts


def test_fixed_pair_found_in_either_order_and_not_reselected(tmp_path):
    labels = np.repeat(['a', 'b'], 10)
    x = np.zeros((20, 2))
    x[10:, 0] = 1  # A stronger but unrequested coordinate.
    bank = {'instance':np.tile(np.arange(10), 2), 'y':labels, 'X':x,
            'pairs':np.array([['p', 'q'], ['r', 's']])}
    config = {'contrasts':[{'classes':['a', 'b'], 'spis':['s', 'r'], 'label':'demo'}]}
    selected = select_contrasts(bank, tmp_path, config)
    assert selected.feature_index.tolist() == [1]
    assert selected.training_auc_oriented.tolist() == [.5]
    with pytest.raises(ValueError):
        select_contrasts(bank, tmp_path, {'contrasts':[{'classes':['a', 'b'], 'spis':['p', 's'], 'label':''}]})


def test_marker_area_is_proportional_to_m():
    np.testing.assert_allclose(marker_area([8, 16, 32]) / marker_area(8), [1, 2, 4])


def test_raincloud_keeps_markers_aligned_when_values_are_missing():
    fig, ax = plt.subplots()
    raincloud(ax, [1, np.nan, 3, 4], 0, 'black', np.random.default_rng(0),
              point_sizes=[8, 16, 32, 8], point_markers=['o', 's', '^', 'o'])
    from matplotlib.collections import PathCollection
    circles, triangles = [c for c in ax.collections if isinstance(c, PathCollection)]
    np.testing.assert_array_equal(circles.get_offsets()[:, 1], [1, 4])
    np.testing.assert_array_equal(triangles.get_offsets()[:, 1], [3])
    np.testing.assert_array_equal(triangles.get_sizes(), [32])
    plt.close(fig)


def test_frozen_config_is_consistent():
    config = load_config()
    names = [c['name'] for c in config['classes']]
    assert len(names) == len(set(names))
    for contrast in config['contrasts']:
        assert set(contrast['classes']) <= set(names) and len(contrast['spis']) == 2
    assert set(T_MARKERS) == {500, 1000, 2000}
