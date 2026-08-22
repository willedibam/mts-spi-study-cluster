#!/usr/bin/env python3
"""Target-blind validation of a frozen T=2000 kinetic-Ising SPI--SPI PC1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_spin_feature_scout import _pc1_master_stability  # noqa: E402
from src.order_parameter_analysis import fit_frozen_pc1  # noqa: E402
from src.order_parameter_features import (  # noqa: E402
    build_meta_feature_matrix,
    stable_spi_names,
    validate_spi_catalogs,
)
from src.utils import load_json  # noqa: E402


EXPECTED_DEVELOPMENT = 72
EXPECTED_VALIDATION = 192
N_BOOTSTRAPS = 500
GATES = {
    "coordinate": 0.90,
    "geometry": 0.85,
    "row": 0.90,
    "pc1_loading": 0.90,
    "pc1_coordinate": 0.90,
    "p99_missingness": 0.01,
    "maximum_missingness": 0.10,
    "safe_core_coordinate": 0.95,
}


def _rho(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3 or np.unique(first[valid]).size < 2 or np.unique(second[valid]).size < 2:
        return float("nan")
    return float(spearmanr(first[valid], second[valid]).statistic)


def _records(data_dir: Path, *, development: bool) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(data_dir.glob("*/*/meta.json")):
        meta = load_json(meta_path)
        params = meta["generator"]["resolved_params"]
        if development and not (int(meta["M"]) == 20 and int(meta["T"]) == 2000):
            continue
        burn = int(params["kinetic_burn_sweeps"])
        rows.append(
            {
                "path": meta_path.parent,
                "physical_path": (
                    "isotropic" if np.isclose(float(params["J_y"]), 1.0) else "anisotropic"
                ),
                "control": float(params["reduced_coupling"]),
                "instance": int(meta["instance_index"]),
                "seed": int(meta["generator"]["seed"]),
                "seed_group_id": str(meta["sampling_design"]["seed_group_id"]),
                "block": 0 if development else (1 if burn == 0 else 2),
                "burn": burn,
                "config_sha256": str(meta["pyspi"]["config_sha256"]),
                "computation_version": str(meta["pyspi"]["version"]["computation"]),
                "n_errors": len(meta["pyspi"].get("errors", {})),
            }
        )
    frame = pd.DataFrame(rows)
    expected = EXPECTED_DEVELOPMENT if development else EXPECTED_VALIDATION
    if len(frame) != expected:
        kind = "development" if development else "validation"
        raise RuntimeError(f"incomplete {kind} set: {len(frame)}/{expected}")
    return frame


def _validate_design(development: pd.DataFrame, validation: pd.DataFrame) -> None:
    combined = pd.concat([development, validation], ignore_index=True)
    if combined["config_sha256"].nunique() != 1 or combined["computation_version"].nunique() != 1:
        raise RuntimeError("pyspi configuration/version mismatch")
    if set(validation["block"]) != {1, 2} or set(validation["burn"]) != {0, 4000}:
        raise RuntimeError("validation blocks do not use the frozen sweep offsets")
    keys = ["physical_path", "control", "instance"]
    first = validation[validation["block"] == 1].sort_values(keys)
    second = validation[validation["block"] == 2].sort_values(keys)
    if list(map(tuple, first[keys].to_numpy())) != list(map(tuple, second[keys].to_numpy())):
        raise RuntimeError("validation blocks are not aligned")
    if not np.array_equal(first["seed"].to_numpy(), second["seed"].to_numpy()):
        raise RuntimeError("validation blocks do not share master seeds")
    for instance, group in validation.groupby("instance"):
        if group["seed"].nunique() != 1 or group["seed_group_id"].nunique() != 1:
            raise RuntimeError(f"validation master {instance} is not paired")


def _imputed_selected(matrix: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(matrix[:, model.feature_indices], dtype=np.float64).copy()
    missing = ~np.isfinite(selected)
    if np.any(missing):
        rows, columns = np.where(missing)
        selected[rows, columns] = model.impute_values[columns]
    return selected, missing.mean(axis=1)


def _block_metrics(
    validation: pd.DataFrame, selected: np.ndarray, coordinate: np.ndarray
) -> tuple[dict[str, float], np.ndarray]:
    keys = ["physical_path", "control", "instance"]
    first = validation[validation["block"] == 1].sort_values(keys)
    second = validation[validation["block"] == 2].sort_values(keys)
    first_indices = first.index.to_numpy()
    second_indices = second.index.to_numpy()
    row_correlations = np.asarray(
        [
            np.corrcoef(selected[i], selected[j])[0, 1]
            for i, j in zip(first_indices, second_indices, strict=True)
        ]
    )

    path_values = []
    for physical_path in np.unique(first["physical_path"]):
        mask = first["physical_path"].to_numpy() == physical_path
        path_values.append(
            (
                _rho(coordinate[first_indices][mask], coordinate[second_indices][mask]),
                _rho(
                    pdist(selected[first_indices][mask]),
                    pdist(selected[second_indices][mask]),
                ),
                float(np.nanmedian(row_correlations[mask])),
            )
        )
    point = np.nanmin(np.asarray(path_values), axis=0)

    rng = np.random.default_rng(480731)
    instances = first["instance"].to_numpy()
    paths = first["physical_path"].to_numpy()
    masters = np.unique(instances)
    draws = np.empty((N_BOOTSTRAPS, 3), dtype=np.float64)
    for draw in range(N_BOOTSTRAPS):
        sampled = rng.choice(masters, size=masters.size, replace=True)
        positions = np.concatenate([np.flatnonzero(instances == master) for master in sampled])
        values = []
        for physical_path in np.unique(paths):
            chosen = positions[paths[positions] == physical_path]
            values.append(
                (
                    _rho(
                        coordinate[first_indices][chosen],
                        coordinate[second_indices][chosen],
                    ),
                    _rho(
                        pdist(selected[first_indices][chosen]),
                        pdist(selected[second_indices][chosen]),
                    ),
                    float(np.nanmedian(row_correlations[chosen])),
                )
            )
        draws[draw] = np.nanmin(np.asarray(values), axis=0)
    lower = np.nanquantile(draws, 0.05, axis=0)
    return (
        {
            "minimum_path_coordinate_spearman": float(point[0]),
            "minimum_path_geometry_spearman": float(point[1]),
            "minimum_path_median_row_correlation": float(point[2]),
            "bootstrap_p05_minimum_path_coordinate_spearman": float(lower[0]),
            "bootstrap_p05_minimum_path_geometry_spearman": float(lower[1]),
            "bootstrap_p05_minimum_path_median_row_correlation": float(lower[2]),
        },
        row_correlations,
    )


def _block_pc1_stability(selected: np.ndarray, validation: pd.DataFrame, model) -> dict[str, float]:
    components = []
    coordinate_agreement = []
    for block in (1, 2):
        indices = validation.index[validation["block"] == block].to_numpy()
        pca = PCA(n_components=1, svd_solver="full").fit(selected[indices])
        component = pca.components_[0]
        cosine = float(np.dot(component, model.component))
        if cosine < 0.0:
            component = -component
            cosine = -cosine
        components.append(component)
        fitted_coordinate = (selected[indices] - pca.mean_) @ component
        frozen_coordinate = (selected[indices] - model.center) @ model.component
        coordinate_agreement.append(_rho(fitted_coordinate, frozen_coordinate))
    return {
        "minimum_development_loading_cosine": float(
            min(np.dot(components[0], model.component), np.dot(components[1], model.component))
        ),
        "between_block_loading_cosine": float(np.dot(components[0], components[1])),
        "minimum_frozen_coordinate_spearman": float(min(coordinate_agreement)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kinetic_ising_feature_scout",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kinetic_ising_t2000_validation",
    )
    args = parser.parse_args()

    development = _records(args.development_dir, development=True)
    validation = _records(args.validation_dir, development=False)
    _validate_design(development, validation)
    all_dirs = development["path"].tolist() + validation["path"].tolist()
    catalog = validate_spi_catalogs(all_dirs)
    frozen_spis = [
        line.strip()
        for line in (args.development_dir / "stable_spis.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    matrix, pairs = build_meta_feature_matrix(all_dirs, catalog, frozen_spis, metric="pearson")
    n_development = len(development)
    development_matrix = matrix[:n_development]
    validation_matrix = matrix[n_development:]
    model = fit_frozen_pc1(development_matrix, variance_threshold=0.05)
    development_coordinate, _ = model.transform(development_matrix)
    validation_coordinate, row_missingness = model.transform(validation_matrix)
    selected, _ = _imputed_selected(validation_matrix, model)
    block_metrics, row_correlations = _block_metrics(
        validation, selected, validation_coordinate
    )
    block_pc1 = _block_pc1_stability(selected, validation, model)
    development_stability = _pc1_master_stability(
        development_matrix[:, model.feature_indices],
        development["instance"].to_numpy(),
        model.component,
        development_coordinate,
    )

    validation_stable, _ = stable_spi_names(
        validation["path"].tolist(), catalog, min_valid_fraction=1.0
    )
    safe_spis = [name for name in frozen_spis if name in set(validation_stable)]
    if safe_spis == frozen_spis:
        safe_coordinate_agreement = 1.0
    else:
        safe_matrix, _ = build_meta_feature_matrix(all_dirs, catalog, safe_spis, metric="pearson")
        safe_model = fit_frozen_pc1(safe_matrix[:n_development], variance_threshold=0.05)
        safe_coordinate, _ = safe_model.transform(safe_matrix[n_development:])
        safe_coordinate_agreement = abs(_rho(validation_coordinate, safe_coordinate))

    gate_results = {
        "block_coordinate": (
            block_metrics["bootstrap_p05_minimum_path_coordinate_spearman"]
            >= GATES["coordinate"]
        ),
        "block_geometry": (
            block_metrics["bootstrap_p05_minimum_path_geometry_spearman"]
            >= GATES["geometry"]
        ),
        "block_row": (
            block_metrics["bootstrap_p05_minimum_path_median_row_correlation"]
            >= GATES["row"]
        ),
        "development_bootstrap_loading": (
            development_stability["bootstrap_p05_loading_cosine"] >= GATES["pc1_loading"]
        ),
        "development_bootstrap_coordinate": (
            development_stability["bootstrap_p05_coordinate_spearman"]
            >= GATES["pc1_coordinate"]
        ),
        "development_loo_loading": (
            development_stability["loo_minimum_loading_cosine"] >= GATES["pc1_loading"]
        ),
        "development_loo_coordinate": (
            development_stability["loo_minimum_coordinate_spearman"]
            >= GATES["pc1_coordinate"]
        ),
        "validation_loading": (
            min(
                block_pc1["minimum_development_loading_cosine"],
                block_pc1["between_block_loading_cosine"],
            )
            >= GATES["pc1_loading"]
        ),
        "validation_coordinate": (
            block_pc1["minimum_frozen_coordinate_spearman"] >= GATES["pc1_coordinate"]
        ),
        "p99_missingness": float(np.quantile(row_missingness, 0.99))
        <= GATES["p99_missingness"],
        "maximum_missingness": float(np.max(row_missingness))
        <= GATES["maximum_missingness"],
        "safe_core_coordinate": safe_coordinate_agreement >= GATES["safe_core_coordinate"],
    }
    passed = all(gate_results.values())
    summary = {
        "passed": passed,
        "development_datasets": len(development),
        "validation_datasets": len(validation),
        "frozen_spis": len(frozen_spis),
        "validation_zero_failure_spis": len(validation_stable),
        "safe_core_spis": len(safe_spis),
        "meta_features": len(pairs),
        "pca_features_retained": int(model.feature_indices.size),
        "development_pc1_variance": model.explained_variance_ratio,
        "development_pc1_stability": development_stability,
        "block_pc1_stability": block_pc1,
        "block_metrics": block_metrics,
        "median_paired_row_correlation": float(np.nanmedian(row_correlations)),
        "p99_row_missingness": float(np.quantile(row_missingness, 0.99)),
        "maximum_row_missingness": float(np.max(row_missingness)),
        "safe_core_coordinate_spearman": safe_coordinate_agreement,
        "maximum_errors_per_dataset": int(validation["n_errors"].max()),
        "gates": GATES,
        "gate_results": gate_results,
        "note": (
            "No order-parameter values were read; SPI selection, T=2000 PC1 fitting "
            "and every eligibility decision are target-blind."
        ),
    }
    args.validation_dir.mkdir(parents=True, exist_ok=True)
    (args.validation_dir / "t2000_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.validation_dir / "t2000_frozen_pc1.npz",
        feature_indices=model.feature_indices,
        impute_values=model.impute_values,
        center=model.center,
        component=model.component,
        explained_variance_ratio=model.explained_variance_ratio,
        frozen_spis=np.asarray(frozen_spis, dtype=str),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
