#!/usr/bin/env python3
"""Compare frozen Kuramoto PC1 before/after independent channel shifts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.order_parameter_analysis import FrozenPC1, residualize_by_group, safe_spearman  # noqa: E402
from src.order_parameter_features import build_meta_feature_matrix, validate_spi_catalogs  # noqa: E402
from src.utils import load_json  # noqa: E402


def _records(data_dir: Path, expected_count: int = 192) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(data_dir.glob("*/*/meta.json")):
        meta = load_json(meta_path)
        rows.append(
            {
                "path": meta_path.parent,
                "distribution": str(meta["source_distribution"]),
                "kappa": float(meta["source_kappa"]),
                "instance": int(meta["source_instance"]),
                "cluster": str(meta["source_seed_group_id"]),
                "outcomes_read": bool(meta["outcomes_copied_or_read"]),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != expected_count or frame["outcomes_read"].any():
        raise RuntimeError("shift-control bank is incomplete or contains outcome leakage")
    frame["kappa_group"] = frame["kappa"].round(6).astype(str)
    return frame.reset_index(drop=True)


def _bootstrap_improvement(
    raw: np.ndarray,
    shifted: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    clusters: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    unique = np.unique(clusters)
    rng = np.random.default_rng(seed)
    draws = np.empty((2000, 2), dtype=np.float64)
    for draw in range(draws.shape[0]):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        selected = np.concatenate([np.flatnonzero(clusters == item) for item in sampled])
        draws[draw, 0] = abs(safe_spearman(raw[selected], target[selected])) - abs(
            safe_spearman(shifted[selected], target[selected])
        )
        raw_within = residualize_by_group(raw[selected], groups[selected])
        shifted_within = residualize_by_group(shifted[selected], groups[selected])
        target_within = residualize_by_group(target[selected], groups[selected])
        draws[draw, 1] = abs(safe_spearman(raw_within, target_within)) - abs(
            safe_spearman(shifted_within, target_within)
        )
    return {
        "raw_overall_absolute_spearman": abs(safe_spearman(raw, target)),
        "shifted_overall_absolute_spearman": abs(safe_spearman(shifted, target)),
        "raw_within_kappa_absolute_spearman": abs(
            safe_spearman(residualize_by_group(raw, groups), residualize_by_group(target, groups))
        ),
        "shifted_within_kappa_absolute_spearman": abs(
            safe_spearman(
                residualize_by_group(shifted, groups), residualize_by_group(target, groups)
            )
        ),
        "raw_minus_shifted_overall_absolute_spearman_ci95": np.quantile(
            draws[:, 0], [0.025, 0.975]
        ).tolist(),
        "raw_minus_shifted_within_kappa_absolute_spearman_ci95": np.quantile(
            draws[:, 1], [0.025, 0.975]
        ).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shift-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_confirmation_shifted",
    )
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_confirmation_contract",
    )
    parser.add_argument("--expected-count", type=int, default=192)
    args = parser.parse_args()

    frame = _records(args.shift_dir, expected_count=args.expected_count)
    archive = np.load(args.contract_dir / "representation_model.npz", allow_pickle=False)
    core_spis = archive["core_spis"].astype(str).tolist()
    catalog = validate_spi_catalogs(frame["path"].tolist())
    matrix, _ = build_meta_feature_matrix(frame["path"].tolist(), catalog, core_spis)
    model = FrozenPC1(
        feature_indices=archive["pc_feature_indices"],
        impute_values=archive["pc_impute_values"],
        center=archive["pc_center"],
        component=archive["pc_component"],
        explained_variance_ratio=float(archive["pc_explained_variance_ratio"]),
    )
    shifted_coordinate, missingness = model.transform(matrix)

    confirmation = np.load(args.contract_dir / "confirmation_results.npz", allow_pickle=False)
    rows = []
    for index in range(confirmation["kappa"].size):
        if (
            confirmation["frequency_sampling"][index] == "random"
            and confirmation["design"][index] == "paired"
            and int(confirmation["instance"][index]) < 8
        ):
            rows.append(
                (
                    str(confirmation["distribution"][index]),
                    round(float(confirmation["kappa"][index]), 6),
                    int(confirmation["instance"][index]),
                    float(confirmation["coordinate_pc1"][index]),
                    float(confirmation["target_full_future_R"][index]),
                )
            )
    lookup = {(dist, kappa, instance): (coordinate, target) for dist, kappa, instance, coordinate, target in rows}
    raw_coordinate = np.empty(len(frame))
    target = np.empty(len(frame))
    for row, item in frame.iterrows():
        key = (item["distribution"], round(float(item["kappa"]), 6), int(item["instance"]))
        raw_coordinate[row], target[row] = lookup[key]

    comparisons = {}
    for index, distribution in enumerate(("gaussian", "logistic")):
        mask = frame["distribution"].eq(distribution).to_numpy()
        comparisons[distribution] = _bootstrap_improvement(
            raw_coordinate[mask],
            shifted_coordinate[mask],
            target[mask],
            frame.loc[mask, "kappa_group"].to_numpy(),
            frame.loc[mask, "cluster"].to_numpy(),
            seed=7300 + index,
        )
    summary = {
        "status": "sensitivity_only",
        "rows": len(frame),
        "all_frozen_meta_features_finite": bool(np.isfinite(matrix).all()),
        "maximum_selected_feature_missingness": float(np.max(missingness)),
        "comparisons": comparisons,
        "interpretation": (
            "A loss after shifting supports collective temporal alignment as a carrier of the "
            "order coordinate. A retained association is not a failure because channel marginals "
            "can legitimately contain synchronization information."
        ),
    }
    (args.contract_dir / "shift_control_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.contract_dir / "shift_control_results.npz",
        distribution=frame["distribution"].to_numpy(dtype=str),
        kappa=frame["kappa"].to_numpy(dtype=np.float64),
        instance=frame["instance"].to_numpy(dtype=np.int64),
        raw_coordinate=raw_coordinate,
        shifted_coordinate=shifted_coordinate,
        target_full_future_R=target,
        selected_feature_missingness=missingness,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
