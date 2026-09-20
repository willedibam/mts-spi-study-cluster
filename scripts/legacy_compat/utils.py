"""Small compatibility helpers replacing the stateful old ``plotter`` class."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.corpus_visualization import plot_mts_heatmap


def save_figure(fig: plt.Figure, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", transparent=True)


def load_geometry(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {name: archive[name] for name in archive.files}


def legacy_raster(values: np.ndarray, *, title: str | None = None):
    return plot_mts_heatmap(values, method="legacy", title=title)
