"""Plot the clusters produced by legacy 2-D HDBSCAN."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from scripts.legacy_compat.utils import load_geometry, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = load_geometry(args.coordinates)
    coordinates = result["coordinates"]
    labels = result["clusters"]
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, s=10, cmap="tab20")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="Legacy HDBSCAN clusters")
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()
