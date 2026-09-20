import numpy as np

from scripts.plot_pair_sampling_coordinate_gallery import sample_examples


def test_gallery_matches_selected_coordinates_of_full_catalogue():
    rng = np.random.default_rng(7)
    matrices = rng.normal(size=(5, 6, 6))
    for a in matrices:
        np.fill_diagonal(a, np.nan)
    full_gold, full = sample_examples(matrices, [3, 8, 15], seed=4, repeats=4)
    selected_gold, selected = sample_examples(matrices[[0, 2]], [3, 8, 15], seed=4, repeats=4)
    np.testing.assert_allclose(selected_gold[0], full_gold[1], rtol=0, atol=2e-7)
    np.testing.assert_allclose(selected[:, :, 0], full[:, :, 1], rtol=0, atol=2e-7)
    np.testing.assert_allclose(full[-1], np.broadcast_to(full_gold, (4, 10)), atol=0)


def test_covariance_is_pearson_for_variance_one_prepared_inputs():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(1000, 5))
    x = (x-x.mean(axis=0))/x.std(axis=0)
    np.testing.assert_allclose(np.cov(x, rowvar=False, bias=True), np.corrcoef(x, rowvar=False), atol=1e-14)


def test_all_draw_paths_use_requested_opacity():
    import matplotlib.pyplot as plt
    from scripts.plot_large_m_pair_sampling import draw_paths
    fig, ax = plt.subplots()
    draw_paths(ax, np.array([8, 16, 32]), np.zeros((3, 128)))
    assert len(ax.lines) == 128
    assert all(line.get_alpha() == .05 for line in ax.lines)
    plt.close(fig)
