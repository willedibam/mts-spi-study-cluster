from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.corpus_visualization import plot_embedding, process_by_time, scale_timeseries


def test_process_by_time_handles_repository_orientations() -> None:
    assert process_by_time(np.zeros((5, 100))).shape == (5, 100)
    assert process_by_time(np.zeros((100, 5))).shape == (5, 100)


def test_heatmap_scaling_is_finite_and_symmetric() -> None:
    values = np.arange(40, dtype=float).reshape(4, 10)
    for method in ("legacy", "zscore", "robust"):
        scaled, limit = scale_timeseries(values, method)
        assert scaled.shape == values.shape
        assert np.isfinite(scaled).all()
        assert limit > 0


def test_neutral_embedding_with_cluster_overlays() -> None:
    values = np.vstack((np.zeros((5, 2)), np.ones((5, 2))))
    fig, ax = plot_embedding(values, clusters=[0] * 5 + [1] * 5)
    assert len(ax.collections) == 1
    assert len(ax.patches) == 2
    plt.close(fig)
