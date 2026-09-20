"""Adaptation of old/plot_dataspace.py: t-SNE plus HDBSCAN in 2-D."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE

from scripts.legacy_compat.utils import load_geometry, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--perplexity", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    geometry = load_geometry(args.geometry)
    values = np.asarray(geometry["transformed"], dtype=float)
    coordinates = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=args.perplexity,
        random_state=args.seed,
    ).fit_transform(values)
    labels = HDBSCAN(min_cluster_size=5).fit_predict(coordinates)
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, s=10, cmap="tab20")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="Legacy 2-D clustering")
    save_figure(fig, args.output)
    coordinate_path = Path(args.output).with_suffix(".npz")
    np.savez_compressed(
        coordinate_path,
        names=geometry["names"],
        coordinates=coordinates.astype(np.float32),
        clusters=labels,
        note=np.asarray("HDBSCAN was fitted to t-SNE coordinates, as in old/"),
    )


if __name__ == "__main__":
    main()
