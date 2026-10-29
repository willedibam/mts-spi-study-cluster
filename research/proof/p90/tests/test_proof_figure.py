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


def test_raincloud_outline_is_opt_in():
    from matplotlib.collections import PolyCollection
    for outline, edged in [(False, False), (True, True)]:
        fig, ax = plt.subplots()
        raincloud(ax, [1., 2., 3., 5.], 0, 'red', np.random.default_rng(0), box=True, outline=outline)
        body = [c for c in ax.collections if isinstance(c, PolyCollection)][0]
        assert (body.get_edgecolor().size > 0 and body.get_edgecolor()[0][3] > 0) == edged
        plt.close(fig)


def test_orientation_is_an_isometry():
    from research.proof.p90.proof_figure import _orient
    xy = np.random.default_rng(0).normal(size=(20, 2))
    d = lambda q: np.linalg.norm(q[:, None] - q[None], axis=-1)
    for angle, reflect in [(0, False), (60, True), (282, False)]:
        np.testing.assert_allclose(d(_orient(xy, angle, reflect)), d(xy), atol=1e-12)


def test_segment_crossing():
    from research.proof.p90.proof_figure import _crosses
    assert _crosses((0, 0), (1, 1), (0, 1), (1, 0))
    assert not _crosses((0, 0), (1, 0), (0, 1), (1, 1))


def test_frozen_layout_places_every_variant_clear_of_the_embedding():
    from research.proof.p90.proof_figure import SILO_W, _geometry, _members
    config = load_config(); a = config['panel_a']
    g = _geometry(config, a['columns'], legend=False, order=a.get('variant_order'))
    assert set(g['anchors']) == {c['name'] for c in config['classes']}
    assert sorted(f for side in a['columns'].values() for f in side) == sorted({c['family'] for c in config['classes']})
    assert SILO_W < g['ex0'] and g['ex0'] + g['side'] < g['W'] - SILO_W
    for f, order in (a.get('variant_order') or {}).items():
        assert [c['name'] for c in _members(config, f, a['variant_order'])] == order


def test_carpet_views_follow_the_frozen_config():
    from research.proof.p90.proof_figure import _carpet_views
    cfg = load_config()['panel_a']['carpets']
    assert [v[:2] for v in _carpet_views('sti-i', 8, 2000, cfg)] == [('_stride12', 12)]
    (_, s0, n), (suffix, stride, _) = _carpet_views('kuramoto_omega-fast', 16, 1000, cfg)
    assert s0 == 1 and suffix == '_whole' and stride*n >= 1000 > (stride-1)*n
    assert [v[:2] for v in _carpet_views('gaussian-noise', 32, 500, cfg)] == [('', 1)]
