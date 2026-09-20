"""Adaptation of old/compute_distance_matrix.py for versioned NPZ features."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.corpus_geometry import condensed_distances, legacy_geometry, load_feature_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", action="append", required=True)
    parser.add_argument("--matrix-key", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-validity", type=float, default=0.90)
    parser.add_argument("--row-validity", type=float, default=0.80)
    args = parser.parse_args()

    values, metadata = load_feature_rows(args.features, args.matrix_key)
    geometry = legacy_geometry(
        values,
        minimum_feature_valid_fraction=args.feature_validity,
        minimum_row_valid_fraction=args.row_validity,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        names=metadata["y"][geometry.row_mask],
        transformed=geometry.values.astype(np.float32),
        distances=condensed_distances(geometry.values).astype(np.float32),
        row_mask=geometry.row_mask,
        feature_mask=geometry.feature_mask,
        feature_mean=geometry.feature_mean.astype(np.float32),
        feature_scale=geometry.feature_scale.astype(np.float32),
        recipe=np.asarray("legacy_zscore_zero_fill"),
    )
    print(
        f"wrote {output}: {geometry.values.shape[0]} datasets x "
        f"{geometry.values.shape[1]} retained features"
    )


if __name__ == "__main__":
    main()
