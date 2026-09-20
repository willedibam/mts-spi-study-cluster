"""Adaptation of the old five-class UMAP proof for current feature artifacts."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from umap import UMAP

from scripts.legacy_compat.utils import save_figure
from src.corpus_geometry import legacy_geometry, load_feature_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", action="append", required=True)
    parser.add_argument("--matrix-key", default="X_sym")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    values, metadata = load_feature_rows(args.features, args.matrix_key)
    geometry = legacy_geometry(values)
    coordinates = UMAP(
        n_components=2,
        n_neighbors=9,
        min_dist=0.75,
        random_state=42,
        n_jobs=1,
    ).fit_transform(geometry.values)
    labels = np.asarray(metadata["y"])[geometry.row_mask].astype(str)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sns.scatterplot(x=coordinates[:, 0], y=coordinates[:, 1], hue=labels, s=15, ax=ax)
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title="Legacy proof-of-principle recipe")
    ax.legend(fontsize=6, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()
