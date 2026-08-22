#!/usr/bin/env python3
"""Freeze a target-free Kuramoto SPI--SPI confirmation representation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diffusion_map import fit_diffusion_map  # noqa: E402
from src.order_parameter_analysis import FrozenPC1  # noqa: E402
from src.order_parameter_features import (  # noqa: E402
    build_meta_feature_matrix,
    validate_spi_catalogs,
)
from src.utils import load_json  # noqa: E402


EXPECTED = {
    "kuramoto-gaussian-paired": 320,
    "kuramoto-gaussian-cell": 80,
    "kuramoto-logistic-paired": 384,
    "kuramoto-logistic-cell": 96,
}
N_PC_BOOTSTRAPS = 50
N_DM_BOOTSTRAPS = 20
GATES = {
    "pc_bootstrap_loading_p05": 0.95,
    "pc_bootstrap_coordinate_p05": 0.95,
    "pc_leave_group_loading_min": 0.90,
    "pc_leave_group_coordinate_min": 0.90,
    "diffusion_bootstrap_coordinate_p05": 0.90,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rho(first: np.ndarray, second: np.ndarray) -> float:
    return float(spearmanr(first, second).statistic)


def _records(data_dir: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(data_dir.glob("*/*/meta.json")):
        meta = load_json(meta_path)
        params = meta["generator"]["resolved_params"]
        control = meta["generator"]["control"]
        sampling = meta["sampling_design"]
        class_name = str(meta["mts_class"])
        design = "paired" if sampling["role"] == "paired-control-path" else "cell"
        rows.append(
            {
                "path": meta_path.parent,
                "class_name": class_name,
                "distribution": str(params["frequency_distribution"]),
                "design": design,
                "kappa": float(control["reduced_value"]),
                "instance": int(meta["instance_index"]),
                "seed_group_id": str(sampling["seed_group_id"]),
                "config_sha256": str(meta["pyspi"]["config_sha256"]),
                "computation_version": str(meta["pyspi"]["version"]["computation"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.groupby("class_name").size().to_dict() != EXPECTED:
        raise RuntimeError("development bank is incomplete or has unexpected classes")
    if frame["config_sha256"].nunique() != 1 or frame["computation_version"].nunique() != 1:
        raise RuntimeError("development pyspi configuration/version mismatch")
    frame["cluster"] = np.where(
        frame["design"].eq("paired"),
        frame["class_name"] + ":" + frame["instance"].astype(str),
        frame["class_name"] + ":" + frame["kappa"].astype(str) + ":" + frame["instance"].astype(str),
    )
    frame["control_group"] = frame["distribution"] + ":" + frame["kappa"].round(6).astype(str)
    return frame.reset_index(drop=True)


def _fit_pc1(values: np.ndarray, *, random_state: int) -> FrozenPC1:
    finite = np.isfinite(values).all(axis=0)
    varying = np.std(values, axis=0) >= 0.05
    feature_indices = np.flatnonzero(finite & varying)
    selected = values[:, feature_indices]
    pca = PCA(
        n_components=1,
        svd_solver="randomized",
        iterated_power=7,
        random_state=int(random_state),
    ).fit(selected)
    component = pca.components_[0].copy()
    if component[np.argmax(np.abs(component))] < 0.0:
        component *= -1.0
    return FrozenPC1(
        feature_indices=feature_indices,
        impute_values=np.median(selected, axis=0),
        center=pca.mean_.copy(),
        component=component,
        explained_variance_ratio=float(pca.explained_variance_ratio_[0]),
    )


def _compare_pc1(values: np.ndarray, indices: np.ndarray, reference: FrozenPC1, q: np.ndarray, seed: int) -> tuple[float, float]:
    selected = values[:, reference.feature_indices]
    pca = PCA(
        n_components=1,
        svd_solver="randomized",
        iterated_power=5,
        random_state=int(seed),
    ).fit(selected[indices])
    component = pca.components_[0]
    loading = float(np.dot(component, reference.component))
    if loading < 0.0:
        component = -component
        loading = -loading
    coordinate = (selected - pca.mean_) @ component
    return loading, abs(_rho(coordinate, q))


def _pc_stability(values: np.ndarray, frame: pd.DataFrame, model: FrozenPC1, q: np.ndarray) -> dict[str, float]:
    clusters = frame["cluster"].to_numpy()
    unique_clusters = np.unique(clusters)
    rng = np.random.default_rng(930811)
    draws = np.empty((N_PC_BOOTSTRAPS, 2), dtype=np.float64)
    for draw in range(N_PC_BOOTSTRAPS):
        sampled = rng.choice(unique_clusters, size=unique_clusters.size, replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == item) for item in sampled])
        draws[draw] = _compare_pc1(values, indices, model, q, 1000 + draw)

    leave_groups = []
    for column in ("distribution", "design", "control_group"):
        labels = frame[column].to_numpy()
        for label in np.unique(labels):
            indices = np.flatnonzero(labels != label)
            leave_groups.append(_compare_pc1(values, indices, model, q, len(leave_groups) + 4000))
    leave = np.asarray(leave_groups)
    return {
        "bootstrap_p05_loading_cosine": float(np.quantile(draws[:, 0], 0.05)),
        "bootstrap_p05_coordinate_spearman": float(np.quantile(draws[:, 1], 0.05)),
        "leave_group_minimum_loading_cosine": float(np.min(leave[:, 0])),
        "leave_group_minimum_coordinate_spearman": float(np.min(leave[:, 1])),
        "leave_group_fits": int(leave.shape[0]),
    }


def _diffusion_stability(values: np.ndarray, frame: pd.DataFrame, reference: np.ndarray) -> tuple[dict[str, float], object]:
    model, coordinate = fit_diffusion_map(values, random_state=7129)
    if abs(_rho(coordinate, reference)) > 1.0 + 1e-12:
        raise AssertionError("invalid rank correlation")
    clusters = frame["cluster"].to_numpy()
    unique_clusters = np.unique(clusters)
    rng = np.random.default_rng(440317)
    agreement = np.empty(N_DM_BOOTSTRAPS, dtype=np.float64)
    for draw in range(N_DM_BOOTSTRAPS):
        sampled = rng.choice(unique_clusters, size=unique_clusters.size, replace=True)
        indices = np.concatenate([np.flatnonzero(clusters == item) for item in sampled])
        candidate, _ = fit_diffusion_map(values[indices], random_state=9000 + draw)
        agreement[draw] = abs(_rho(candidate.transform(values), coordinate))
    return (
        {
            "dimension": int(model.pca_components.shape[0]),
            "pca_explained_variance": float(model.explained_variance),
            "neighbours": int(model.neighbours),
            "bandwidth": float(model.bandwidth),
            "eigenvalue": float(model.eigenvalue),
            "bootstrap_p05_coordinate_spearman": float(np.quantile(agreement, 0.05)),
        },
        model,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_order_benchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_confirmation_contract",
    )
    args = parser.parse_args()

    frame = _records(args.data_dir)
    old_model_path = args.data_dir / "benchmark_model.npz"
    old = np.load(old_model_path, allow_pickle=False)
    old_spis = old["stable_spis"].astype(str)
    selected_pairs = old["complete_stable_columns"] & old["no_phase_columns"]
    nodes = np.unique(np.r_[old["pair_left"][selected_pairs], old["pair_right"][selected_pairs]])
    core_spis = old_spis[nodes].tolist()
    if len(core_spis) != 197 or int(selected_pairs.sum()) != 19306:
        raise RuntimeError("the disclosed development core is not the frozen 197-SPI clique")

    catalog = validate_spi_catalogs(frame["path"].tolist())
    matrix, pairs = build_meta_feature_matrix(frame["path"].tolist(), catalog, core_spis)
    if matrix.shape != (880, 19306) or not np.isfinite(matrix).all():
        raise RuntimeError("frozen development core is incomplete or nonfinite")
    model = _fit_pc1(matrix, random_state=8123)
    q, missingness = model.transform(matrix)
    if np.any(missingness):
        raise RuntimeError("development PC1 unexpectedly required imputation")
    pc_stability = _pc_stability(matrix, frame, model, q)

    diffusion_input = matrix[:, model.feature_indices]
    diffusion_stability, diffusion = _diffusion_stability(diffusion_input, frame, q)
    diffusion_available = bool(
        diffusion_stability["bootstrap_p05_coordinate_spearman"]
        >= GATES["diffusion_bootstrap_coordinate_p05"]
    )
    gate_results = {
        "pc_bootstrap_loading": pc_stability["bootstrap_p05_loading_cosine"] >= GATES["pc_bootstrap_loading_p05"],
        "pc_bootstrap_coordinate": pc_stability["bootstrap_p05_coordinate_spearman"] >= GATES["pc_bootstrap_coordinate_p05"],
        "pc_leave_group_loading": pc_stability["leave_group_minimum_loading_cosine"] >= GATES["pc_leave_group_loading_min"],
        "pc_leave_group_coordinate": pc_stability["leave_group_minimum_coordinate_spearman"] >= GATES["pc_leave_group_coordinate_min"],
    }
    pc_eligible = bool(all(gate_results.values()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "representation_model.npz"
    payload = {
        "core_spis": np.asarray(core_spis, dtype=str),
        "pair_left": np.asarray([core_spis.index(left) for left, _ in pairs], dtype=np.int16),
        "pair_right": np.asarray([core_spis.index(right) for _, right in pairs], dtype=np.int16),
        "pc_feature_indices": model.feature_indices,
        "pc_impute_values": model.impute_values,
        "pc_center": model.center,
        "pc_component": model.component,
        "pc_explained_variance_ratio": np.asarray(model.explained_variance_ratio),
        "development_pc1": q,
        "development_class_name": frame["class_name"].to_numpy(dtype=str),
        "development_distribution": frame["distribution"].to_numpy(dtype=str),
        "development_design": frame["design"].to_numpy(dtype=str),
        "development_kappa": frame["kappa"].to_numpy(),
        "development_instance": frame["instance"].to_numpy(dtype=np.int16),
        "dm_available": np.asarray(diffusion_available),
        "dm_pca_mean": diffusion.pca_mean,
        "dm_pca_components": diffusion.pca_components,
        "dm_reference_scores": diffusion.reference_scores,
        "dm_reference_density": diffusion.reference_density,
        "dm_reference_eigenfunction": diffusion.reference_eigenfunction,
        "dm_eigenvalue": np.asarray(diffusion.eigenvalue),
        "dm_bandwidth": np.asarray(diffusion.bandwidth),
    }
    np.savez_compressed(model_path, **payload)
    summary = {
        "status": "eligible" if pc_eligible else "failed",
        "outcomes_read": False,
        "disclosed_development_rows": len(frame),
        "core_spis": len(core_spis),
        "core_meta_features": len(pairs),
        "retained_pc_features": int(model.feature_indices.size),
        "pc_explained_variance_ratio": float(model.explained_variance_ratio),
        "pc_stability": pc_stability,
        "diffusion_stability": diffusion_stability,
        "diffusion_available": diffusion_available,
        "gates": GATES,
        "gate_results": gate_results,
        "old_model_sha256": _sha256(old_model_path),
        "representation_model_sha256": _sha256(model_path),
        "pyspi_config_sha256": frame["config_sha256"].iloc[0],
        "pyspi_computation_version": frame["computation_version"].iloc[0],
    }
    (args.output_dir / "representation_contract.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "core_spis.txt").write_text("\n".join(core_spis) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if pc_eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
