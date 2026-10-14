import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from research.cases.legacy_case_figures import (
    _paired_values, _correlation, apply_plot_style,
    plot_spi_space_individual, plot_spi_space_individual_embed_lags,
)


def test_pair_mask_preserves_lag_alignment(tmp_path):
    a = np.array([[0, 1, np.nan], [3, 0, 5], [np.nan, 7, 0.]])
    b = np.array([[0, 8, 9], [8, 0, 2], [9, 2, 0.]])
    np.savez(tmp_path / "spi_mpis.npz", a=a, b=b)
    (tmp_path / "meta.json").write_text(json.dumps({"generator": {"lags": [1, 4, 9]}}))
    x, y, i, j = _paired_values(tmp_path, ["a", "b"])
    np.testing.assert_equal(x, [2, 6])
    np.testing.assert_equal(y, [8, 2])
    np.testing.assert_equal(np.column_stack([i, j]), [[0, 1], [1, 2]])
    fig, ax = plt.subplots()
    plot_spi_space_individual_embed_lags(tmp_path, ["a", "b"], ax, metric="pearson")
    np.testing.assert_equal(ax.collections[0].get_array(), [3, 5])
    plt.close(fig)


def test_constant_profile_is_undefined_and_embedded_plot_does_not_show(tmp_path, monkeypatch):
    assert np.isnan(_correlation(np.ones(3), np.arange(3), "pearson"))
    np.savez(tmp_path / "spi_mpis.npz", a=np.ones((3, 3)), b=np.eye(3))
    fig, ax = plt.subplots()
    before = plt.get_fignums()
    apply_plot_style()
    assert plt.get_fignums() == before
    def forbidden_show():
        raise AssertionError("An embedded plot must not display an unfinished figure")
    monkeypatch.setattr(plt, "show", forbidden_show)
    plot_spi_space_individual(tmp_path, ["a", "b"], ax=ax, metric="pearson")
    assert len(ax.collections[0].get_offsets()) == 3
    assert "Undefined" in ax.texts[0].get_text()
    plt.close(fig)
