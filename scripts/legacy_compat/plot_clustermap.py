"""Adaptation of old/plot_clustermap.py for an NPZ geometry artifact."""
from __future__ import annotations

import argparse

import numpy as np
import seaborn as sns

from scripts.legacy_compat.utils import load_geometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-features", type=int, default=3000)
    args = parser.parse_args()
    geometry = load_geometry(args.geometry)
    values = np.asarray(geometry["transformed"], dtype=float)
    if values.shape[1] > args.max_features:
        indices = np.linspace(0, values.shape[1] - 1, args.max_features).astype(int)
        values = values[:, indices]
    grid = sns.clustermap(
        values,
        cmap="coolwarm",
        vmin=-2,
        vmax=2,
        xticklabels=False,
        yticklabels=False,
        figsize=(10, 5),
    )
    grid.savefig(args.output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
