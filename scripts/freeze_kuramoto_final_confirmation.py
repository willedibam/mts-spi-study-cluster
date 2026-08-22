#!/usr/bin/env python3
"""Freeze the single target-blind Kuramoto assay redesign."""

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

from scripts.analyze_kuramoto_confirmation import _records as _failed_records  # noqa: E402
from scripts.freeze_kuramoto_confirmation import (  # noqa: E402
    GATES,
    N_PC_BOOTSTRAPS,
    _compare_pc1,
    _diffusion_stability,
    _fit_pc1,
    _records as _old_records,
    _sha256,
)
from src.order_parameter_features import (  # noqa: E402
    build_meta_feature_matrix,
    explicit_phase_spi_names,
    stable_spi_names,
    validate_spi_catalogs,
)


MIN_CORE_SPIS = 100


def _development_frame(old_dir: Path, failed_dir: Path) -> pd.DataFrame:
    old = _old_records(old_dir).copy()
    old["source"] = "disclosed_old"
    old["frequency_sampling"] = "random"
    failed = _failed_records(failed_dir).copy()
    failed["source"] = "blinded_eligibility_null"
    frame = pd.concat([old, failed], ignore_index=True, sort=False)
    frame["cluster"] = frame["source"] + ":" + frame["cluster"].astype(str)
    frame["path_group"] = (
        frame["distribution"].astype(str)
        + ":"
        + frame["frequency_sampling"].astype(str)
        + ":"
        + frame["design"].astype(str)
    )
    frame["control_group"] = frame["kappa"].round(6).astype(str)
    if len(frame) != 2032:
        raise RuntimeError(f"expected 2032 target-free development rows, found {len(frame)}")
    return frame


def _pc_stability(
    values: np.ndarray,
    frame: pd.DataFrame,
    model: object,
    reference: np.ndarray,
) -> dict[str, object]:
    clusters = frame["cluster"].to_numpy()
    unique_clusters = np.unique(clusters)
    rng = np.random.default_rng(631907)
    draws = np.empty((N_PC_BOOTSTRAPS, 2), dtype=np.float64)
    for draw in range(N_PC_BOOTSTRAPS):
        sampled = rng.choice(unique_clusters, size=unique_clusters.size, replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == item) for item in sampled])
        draws[draw] = _compare_pc1(values, indices, model, reference, 11000 + draw)

    leave_rows: list[dict[str, object]] = []
    for column in ("source", "path_group", "control_group"):
        labels = frame[column].to_numpy()
        for label in np.unique(labels):
            indices = np.flatnonzero(labels != label)
            loading, coordinate = _compare_pc1(
                values, indices, model, reference, 13000 + len(leave_rows)
            )
            leave_rows.append(
                {
                    "grouping": column,
                    "left_out": str(label),
                    "loading_cosine": loading,
                    "coordinate_spearman": coordinate,
                }
            )
    return {
        "bootstrap_p05_loading_cosine": float(np.quantile(draws[:, 0], 0.05)),
        "bootstrap_p05_coordinate_spearman": float(np.quantile(draws[:, 1], 0.05)),
        "leave_group_minimum_loading_cosine": float(
            min(row["loading_cosine"] for row in leave_rows)
        ),
        "leave_group_minimum_coordinate_spearman": float(
            min(row["coordinate_spearman"] for row in leave_rows)
        ),
        "leave_group_fits": len(leave_rows),
        "leave_group_results": leave_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_order_benchmark",
    )
    parser.add_argument(
        "--failed-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_confirmation",
    )
    parser.add_argument(
        "--failed-contract-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_confirmation_contract",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_final_confirmation_contract",
    )
    args = parser.parse_args()

    eligibility_path = args.failed_contract_dir / "confirmation_eligibility.json"
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    if eligibility.get("passed") or eligibility.get("outcomes_read"):
        raise RuntimeError("the eligibility-null bank is not target-sealed")
    if (args.failed_contract_dir / "confirmation_summary.json").exists():
        raise RuntimeError("an outcome summary exists for the eligibility-null bank")

    frame = _development_frame(args.old_dir, args.failed_dir)
    if frame["config_sha256"].nunique() != 1:
        raise RuntimeError("development banks do not use the same pyspi configuration")
    catalog = validate_spi_catalogs(frame["path"].tolist())
    excluded = explicit_phase_spi_names(catalog)
    core_spis, validity = stable_spi_names(
        frame["path"].tolist(), catalog, min_valid_fraction=1.0, exclude=excluded
    )
    if len(core_spis) < MIN_CORE_SPIS:
        raise RuntimeError(f"only {len(core_spis)} all-row-stable non-phase SPIs remain")

    matrix, pairs = build_meta_feature_matrix(frame["path"].tolist(), catalog, core_spis)
    expected_features = len(core_spis) * (len(core_spis) - 1) // 2
    if matrix.shape != (len(frame), expected_features) or not np.isfinite(matrix).all():
        raise RuntimeError("the all-row-stable SPI clique is incomplete or nonfinite")

    model = _fit_pc1(matrix, random_state=17239)
    coordinate, missingness = model.transform(matrix)
    if np.any(missingness) or not np.isfinite(coordinate).all():
        raise RuntimeError("redesigned development PC1 required imputation")
    pc_stability = _pc_stability(matrix, frame, model, coordinate)
    diffusion_stability, diffusion = _diffusion_stability(
        matrix[:, model.feature_indices], frame, coordinate
    )
    diffusion_available = bool(
        diffusion_stability["bootstrap_p05_coordinate_spearman"]
        >= GATES["diffusion_bootstrap_coordinate_p05"]
    )
    gate_results = {
        "minimum_core_spis": len(core_spis) >= MIN_CORE_SPIS,
        "zero_development_missingness": bool(np.isfinite(matrix).all()),
        "pc_bootstrap_loading": pc_stability["bootstrap_p05_loading_cosine"]
        >= GATES["pc_bootstrap_loading_p05"],
        "pc_bootstrap_coordinate": pc_stability["bootstrap_p05_coordinate_spearman"]
        >= GATES["pc_bootstrap_coordinate_p05"],
        "pc_leave_group_loading": pc_stability["leave_group_minimum_loading_cosine"]
        >= GATES["pc_leave_group_loading_min"],
        "pc_leave_group_coordinate": pc_stability["leave_group_minimum_coordinate_spearman"]
        >= GATES["pc_leave_group_coordinate_min"],
    }
    eligible = bool(all(gate_results.values()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "representation_model.npz"
    np.savez_compressed(
        model_path,
        core_spis=np.asarray(core_spis, dtype=str),
        pair_left=np.asarray([core_spis.index(left) for left, _ in pairs], dtype=np.int16),
        pair_right=np.asarray([core_spis.index(right) for _, right in pairs], dtype=np.int16),
        pc_feature_indices=model.feature_indices,
        pc_impute_values=model.impute_values,
        pc_center=model.center,
        pc_component=model.component,
        pc_explained_variance_ratio=np.asarray(model.explained_variance_ratio),
        development_pc1=coordinate,
        development_source=frame["source"].to_numpy(dtype=str),
        development_class_name=frame["class_name"].to_numpy(dtype=str),
        development_kappa=frame["kappa"].to_numpy(),
        dm_available=np.asarray(diffusion_available),
        dm_pca_mean=diffusion.pca_mean,
        dm_pca_components=diffusion.pca_components,
        dm_reference_scores=diffusion.reference_scores,
        dm_reference_density=diffusion.reference_density,
        dm_reference_eigenfunction=diffusion.reference_eigenfunction,
        dm_eigenvalue=np.asarray(diffusion.eigenvalue),
        dm_bandwidth=np.asarray(diffusion.bandwidth),
    )
    summary = {
        "status": "eligible" if eligible else "failed",
        "outcomes_read": False,
        "redesign_number": 1,
        "further_redesign_permitted": False,
        "development_rows": len(frame),
        "development_sources": frame.groupby("source").size().to_dict(),
        "core_spis": len(core_spis),
        "excluded_explicit_phase_spis": len(excluded),
        "minimum_core_validity_rate": float(min(validity[name] for name in core_spis)),
        "core_meta_features": len(pairs),
        "retained_pc_features": int(model.feature_indices.size),
        "pc_explained_variance_ratio": float(model.explained_variance_ratio),
        "pc_stability": pc_stability,
        "diffusion_stability": diffusion_stability,
        "diffusion_available": diffusion_available,
        "gates": {**GATES, "minimum_core_spis": MIN_CORE_SPIS},
        "gate_results": gate_results,
        "failed_bank_eligibility_sha256": _sha256(eligibility_path),
        "failed_bank_outcomes_read": False,
        "representation_model_sha256": _sha256(model_path),
        "pyspi_config_sha256": frame["config_sha256"].iloc[0],
        "selection_rule": (
            "All non-explicit-phase SPIs finite and nonconstant on every old plus "
            "eligibility-null X row; all pairwise SPI correlations retained before "
            "the frozen SD>=0.05 PC1 filter. No order parameter was read."
        ),
    }
    (args.output_dir / "representation_contract.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
