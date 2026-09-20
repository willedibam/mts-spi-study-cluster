"""Render one raw MTS using the exact old heatmap scaling orientation."""
from __future__ import annotations

import argparse

import numpy as np

from scripts.legacy_compat.utils import legacy_raster, save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.database, allow_pickle=False) as archive:
        if args.dataset not in archive.files:
            raise KeyError(f"dataset {args.dataset!r} is absent from {args.database}")
        values = archive[args.dataset]
    fig, _ = legacy_raster(values, title=args.dataset)
    save_figure(fig, args.output)


if __name__ == "__main__":
    main()
