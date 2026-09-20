"""List nearest datasets from a legacy or optimized geometry artifact."""
from __future__ import annotations

import argparse

import pandas as pd

from scripts.legacy_compat.utils import load_geometry
from src.corpus_geometry import nearest_neighbour_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--focal", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("-k", type=int, default=20)
    args = parser.parse_args()
    geometry = load_geometry(args.geometry)
    values = geometry.get("query_pca", geometry.get("transformed"))
    names = geometry.get("query_names", geometry.get("names"))
    if values is None or names is None:
        raise KeyError("geometry artifact lacks coordinates or dataset names")
    rows = nearest_neighbour_table(values, names, args.focal, k=args.k)
    pd.DataFrame(rows, columns=("dataset", "distance")).to_csv(args.output, index=False)
    print(pd.DataFrame(rows, columns=("dataset", "distance")).to_string(index=False))


if __name__ == "__main__":
    main()
