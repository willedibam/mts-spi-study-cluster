#!/usr/bin/env python3
"""Apply the frozen Kuramoto representation, then reveal canonical future R."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diffusion_map import FrozenDiffusionMap  # noqa: E402
from src.order_parameter_analysis import (  # noqa: E402
    FrozenPC1,
    clustered_bootstrap_difference,
    clustered_bootstrap_mae,
    clustered_bootstrap_spearman,
    input_only_features,
    safe_spearman,
)
from src.order_parameter_features import (  # noqa: E402
    build_meta_feature_matrix,
    validate_spi_catalogs,
)
from src.process_features import _edge_vectors  # noqa: E402
from src.utils import load_json  # noqa: E402


EXPECTED = {
    "kuramoto-confirm-gaussian-paired": 384,
    "kuramoto-confirm-gaussian-cell": 96,
    "kuramoto-confirm-gaussian-regular": 96,
    "kuramoto-confirm-logistic-paired": 384,
    "kuramoto-confirm-logistic-cell": 96,
    "kuramoto-confirm-logistic-regular": 96,
}
N_BOOTSTRAPS = 2000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(data_dir: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(data_dir.glob("*/*/meta.json")):
        meta = load_json(meta_path)
        params = meta["generator"]["resolved_params"]
        sampling = meta["sampling_design"]
        experiment = meta["experiment"]
        role = str(sampling["role"])
        design = "paired" if role == "paired-control-path" else "cell"
        rows.append(
            {
                "path": meta_path.parent,
                "class_name": str(meta["mts_class"]),
                "distribution": str(params["frequency_distribution"]),
                "frequency_sampling": str(params["frequency_sampling"]),
                "design": design,
                "kappa": float(meta["generator"]["control"]["reduced_value"]),
                "instance": int(meta["instance_index"]),
                "seed": int(meta["generator"]["seed"]),
                "seed_group_id": str(sampling["seed_group_id"]),
                "config_sha256": str(meta["pyspi"]["config_sha256"]),
                "computation_version": str(meta["pyspi"]["version"]["computation"]),
                "experiment_config_sha256": str(experiment["config_sha256"]),
                "generation_git_commit": str(experiment["git_commit"]),
                "generation_git_dirty": bool(experiment["git_dirty"]),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.groupby("class_name").size().to_dict() != EXPECTED:
        raise RuntimeError("confirmation bank is incomplete or has unexpected classes")
    if frame["config_sha256"].nunique() != 1 or frame["computation_version"].nunique() != 1:
        raise RuntimeError("confirmation pyspi configuration/version mismatch")
    if frame["experiment_config_sha256"].nunique() != 1:
        raise RuntimeError("confirmation generation configuration mismatch")
    if frame["generation_git_commit"].nunique() != 1 or frame["generation_git_dirty"].any():
        raise RuntimeError("confirmation generation checkout was not one clean commit")
    frame["kappa_group"] = frame["kappa"].round(6).astype(str)
    frame["cluster"] = frame["class_name"] + ":" + frame["seed_group_id"]
    return frame.reset_index(drop=True)


def _strength_lower(interval: np.ndarray) -> float:
    low, high = map(float, interval)
    return min(abs(low), abs(high)) if low * high > 0.0 else 0.0


def _association_block(
    coordinate: np.ndarray,
    target: np.ndarray,
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    overall, within = clustered_bootstrap_spearman(
        coordinate[mask],
        target[mask],
        frame.loc[mask, "kappa_group"].to_numpy(),
        frame.loc[mask, "cluster"].to_numpy(),
        n_resamples=N_BOOTSTRAPS,
        seed=seed,
    )
    overall_ci = np.quantile(overall, [0.025, 0.975])
    within_ci = np.quantile(within, [0.025, 0.975])
    return {
        "overall_spearman": safe_spearman(coordinate[mask], target[mask]),
        "overall_ci95": overall_ci.tolist(),
        "overall_absolute_ci_lower": _strength_lower(overall_ci),
        "within_kappa_spearman": safe_spearman(
            coordinate[mask] - pd.Series(coordinate[mask]).groupby(frame.loc[mask, "kappa_group"].to_numpy()).transform("mean").to_numpy(),
            target[mask] - pd.Series(target[mask]).groupby(frame.loc[mask, "kappa_group"].to_numpy()).transform("mean").to_numpy(),
        ),
        "within_kappa_ci95": within_ci.tolist(),
        "within_kappa_absolute_ci_lower": _strength_lower(within_ci),
    }


def _lcss_mean(paths: list[Path], catalog: list[dict]) -> np.ndarray:
    directed = {str(item["name"]): bool(item.get("directed", False)) for item in catalog}
    values = np.empty(len(paths), dtype=np.float64)
    for row, path in enumerate(paths):
        with np.load(path / "spi_mpis.npz") as archive:
            vector = _edge_vectors("lcss", archive["lcss"], directed["lcss"], False)[0][1]
        values[row] = float(np.mean(vector))
    return values


def _predict(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.interp(values, x, y, left=y[0], right=y[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_confirmation",
    )
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_confirmation_contract",
    )
    args = parser.parse_args()

    representation_path = args.contract_dir / "representation_model.npz"
    readout_path = args.contract_dir / "readout_model.npz"
    representation_contract = json.loads((args.contract_dir / "representation_contract.json").read_text())
    readout_contract = json.loads((args.contract_dir / "readout_contract.json").read_text())
    if representation_contract["status"] != "eligible" or readout_contract["status"] != "frozen":
        raise RuntimeError("confirmation contracts are not frozen and eligible")
    if _sha256(representation_path) != readout_contract["representation_model_sha256"]:
        raise RuntimeError("frozen representation hash mismatch")
    if _sha256(readout_path) != readout_contract["readout_model_sha256"]:
        raise RuntimeError("frozen readout hash mismatch")

    frame = _records(args.data_dir)
    model_archive = np.load(representation_path, allow_pickle=False)
    core_spis = model_archive["core_spis"].astype(str).tolist()
    catalog = validate_spi_catalogs(frame["path"].tolist())
    matrix, pairs = build_meta_feature_matrix(frame["path"].tolist(), catalog, core_spis)
    if matrix.shape[1] != len(core_spis) * (len(core_spis) - 1) // 2:
        raise RuntimeError("confirmation pair ordering/shape mismatch")
    model = FrozenPC1(
        feature_indices=model_archive["pc_feature_indices"],
        impute_values=model_archive["pc_impute_values"],
        center=model_archive["pc_center"],
        component=model_archive["pc_component"],
        explained_variance_ratio=float(model_archive["pc_explained_variance_ratio"]),
    )
    coordinate, row_missingness = model.transform(matrix)
    eligibility = {
        "outcomes_read": False,
        "rows": len(frame),
        "core_spis": len(core_spis),
        "core_meta_features": len(pairs),
        "all_frozen_meta_features_finite": bool(np.isfinite(matrix).all()),
        "all_coordinates_finite": bool(np.isfinite(coordinate).all()),
        "maximum_selected_feature_missingness": float(np.max(row_missingness)),
        "per_class_maximum_selected_feature_missingness": {
            name: float(np.max(row_missingness[group.index.to_numpy()]))
            for name, group in frame.groupby("class_name")
        },
        "representation_model_sha256": _sha256(representation_path),
        "readout_model_sha256": _sha256(readout_path),
        "confirmation_config_sha256": frame["experiment_config_sha256"].iloc[0],
        "generation_git_commit": frame["generation_git_commit"].iloc[0],
    }
    eligibility["passed"] = bool(
        eligibility["all_frozen_meta_features_finite"]
        and eligibility["all_coordinates_finite"]
    )
    args.contract_dir.mkdir(parents=True, exist_ok=True)
    eligibility_path = args.contract_dir / "confirmation_eligibility.json"
    eligibility_path.write_text(json.dumps(eligibility, indent=2, sort_keys=True) + "\n")
    if not eligibility["passed"]:
        print(json.dumps(eligibility, indent=2, sort_keys=True))
        return 2

    diffusion_coordinate = np.full(len(frame), np.nan)
    if bool(model_archive["dm_available"]):
        diffusion = FrozenDiffusionMap(
            pca_mean=model_archive["dm_pca_mean"],
            pca_components=model_archive["dm_pca_components"],
            reference_scores=model_archive["dm_reference_scores"],
            reference_density=model_archive["dm_reference_density"],
            reference_eigenfunction=model_archive["dm_reference_eigenfunction"],
            eigenvalue=float(model_archive["dm_eigenvalue"]),
            bandwidth=float(model_archive["dm_bandwidth"]),
            neighbours=int(representation_contract["diffusion_stability"]["neighbours"]),
            explained_variance=float(representation_contract["diffusion_stability"]["pca_explained_variance"]),
        )
        diffusion_coordinate = diffusion.transform(matrix[:, model.feature_indices])

    target = np.empty(len(frame), dtype=np.float64)
    complement = np.empty(len(frame), dtype=np.float64)
    for row, path in enumerate(frame["path"]):
        with np.load(path / "ground_truth.npz") as truth:
            target[row] = float(np.mean(truth["r_full_future"]))
            complement[row] = float(np.mean(truth["r_unobserved_future"]))
    eligibility["outcomes_read"] = True
    eligibility_path.write_text(json.dumps(eligibility, indent=2, sort_keys=True) + "\n")

    input_rows = [input_only_features(np.load(path / "timeseries.npy")) for path in frame["path"]]
    baselines = {
        "kappa": frame["kappa"].to_numpy(),
        "mean_abs_correlation": np.asarray([item["mean_abs_correlation"] for item in input_rows]),
        "analytic_phase_coherence": np.asarray([item["analytic_phase_coherence"] for item in input_rows]),
        "temporal_spectral_entropy": np.asarray([item["mean_temporal_spectral_entropy"] for item in input_rows]),
        "development_selected_lcss_mean": _lcss_mean(frame["path"].tolist(), catalog),
    }
    readout = np.load(readout_path, allow_pickle=False)
    prediction = _predict(coordinate, readout["pc1_x"], readout["pc1_y"])
    kappa_prediction = _predict(frame["kappa"].to_numpy(), readout["kappa_x"], readout["kappa_y"])
    intercept_prediction = np.full(len(frame), float(readout["intercept"]))

    associations: dict[str, object] = {}
    baseline_associations: dict[str, object] = {}
    calibration: dict[str, object] = {}
    gate_results: dict[str, bool] = {}
    gates = readout_contract["confirmation_gates"]
    for path_index, distribution in enumerate(("gaussian", "logistic")):
        primary = (
            frame["distribution"].eq(distribution)
            & frame["frequency_sampling"].eq("random")
            & frame["design"].eq("paired")
        ).to_numpy()
        full = _association_block(coordinate, target, frame, primary, seed=100 + path_index)
        hidden = _association_block(coordinate, complement, frame, primary, seed=200 + path_index)
        associations[distribution] = {"full_future_R": full, "hidden_complement_future_R": hidden}
        baseline_associations[distribution] = {
            name: _association_block(values, target, frame, primary, seed=300 + 10 * path_index + index)
            for index, (name, values) in enumerate(baselines.items())
        }
        mae = clustered_bootstrap_mae(
            target[primary], prediction[primary], frame.loc[primary, "cluster"].to_numpy(),
            n_resamples=N_BOOTSTRAPS, seed=500 + path_index,
        )
        difference = clustered_bootstrap_difference(
            target[primary], prediction[primary], intercept_prediction[primary],
            frame.loc[primary, "cluster"].to_numpy(), n_resamples=N_BOOTSTRAPS, seed=600 + path_index,
        )
        calibration[distribution] = {
            "mae": float(np.mean(np.abs(target[primary] - prediction[primary]))),
            "mae_ci95": np.quantile(mae, [0.025, 0.975]).tolist(),
            "pc1_minus_intercept_mae_difference_ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
            "kappa_calibration_mae": float(np.mean(np.abs(target[primary] - kappa_prediction[primary]))),
        }
        gate_results[f"{distribution}_overall"] = bool(
            full["overall_absolute_ci_lower"] >= gates["minimum_overall_absolute_spearman_ci95_lower_each_random_path"]
        )
        gate_results[f"{distribution}_within"] = bool(
            full["within_kappa_absolute_ci_lower"] >= gates["minimum_within_kappa_absolute_spearman_ci95_lower_each_random_path"]
        )
        gate_results[f"{distribution}_hidden_within"] = bool(
            hidden["within_kappa_absolute_ci_lower"] >= gates["minimum_within_kappa_hidden_complement_absolute_spearman_ci95_lower_each_random_path"]
        )
        gate_results[f"{distribution}_numerical_mae"] = bool(
            calibration[distribution]["mae_ci95"][1] <= gates["maximum_calibrated_mae_ci95_upper_each_random_path"]
        )
        gate_results[f"{distribution}_beats_intercept"] = bool(
            calibration[distribution]["pc1_minus_intercept_mae_difference_ci95"][1]
            < gates["calibrated_pc1_minus_intercept_mae_ci95_upper_each_random_path"]
        )

    sensitivities = {}
    for distribution in ("gaussian", "logistic"):
        cell = (
            frame["distribution"].eq(distribution)
            & frame["frequency_sampling"].eq("random")
            & frame["design"].eq("cell")
        ).to_numpy()
        regular = (
            frame["distribution"].eq(distribution)
            & frame["frequency_sampling"].eq("regular")
        ).to_numpy()
        sensitivities[distribution] = {
            "independent_cell": _association_block(coordinate, target, frame, cell, seed=800),
            "regular_frequency": _association_block(coordinate, target, frame, regular, seed=900),
        }

    passed = bool(all(gate_results.values()))
    summary = {
        "status": "passed" if passed else "failed",
        "claim": (
            "A frozen unsupervised SPI--SPI coordinate tracks the changing finite-N Kuramoto "
            "order parameter on untouched data; its numerical q-to-R readout is supervised."
        ),
        "eligibility": eligibility,
        "gates": gates,
        "gate_results": gate_results,
        "associations": associations,
        "calibration": calibration,
        "baseline_associations": baseline_associations,
        "sensitivities": sensitivities,
        "diffusion_map_available": bool(model_archive["dm_available"]),
        "diffusion_map_pc1_spearman": safe_spearman(diffusion_coordinate, coordinate),
    }
    (args.contract_dir / "confirmation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.contract_dir / "confirmation_results.npz",
        class_name=frame["class_name"].to_numpy(dtype=str),
        distribution=frame["distribution"].to_numpy(dtype=str),
        frequency_sampling=frame["frequency_sampling"].to_numpy(dtype=str),
        design=frame["design"].to_numpy(dtype=str),
        kappa=frame["kappa"].to_numpy(),
        instance=frame["instance"].to_numpy(dtype=np.int16),
        coordinate_pc1=coordinate,
        coordinate_diffusion=diffusion_coordinate,
        target_full_future_R=target,
        target_hidden_complement_future_R=complement,
        prediction_R=prediction,
        prediction_kappa_baseline_R=kappa_prediction,
        **{f"baseline_{name}": values for name, values in baselines.items()},
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
