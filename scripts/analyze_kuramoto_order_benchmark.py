#!/usr/bin/env python3
"""Frozen, leakage-safe analysis of the canonical Kuramoto benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.order_parameter_analysis import (  # noqa: E402
    clustered_bootstrap_difference,
    clustered_bootstrap_mae,
    clustered_bootstrap_spearman,
    fit_frozen_pc1,
    input_only_features,
    residualize_by_group,
    safe_spearman,
)
from src.order_parameter_features import (  # noqa: E402
    build_meta_feature_matrix,
    explicit_phase_spi_names,
    stable_spi_names,
    validate_spi_catalogs,
)
from src.process_features import _edge_vectors  # noqa: E402
from src.utils import load_json  # noqa: E402


EXPECTED = {
    "kuramoto-gaussian-paired": 320,
    "kuramoto-gaussian-cell": 80,
    "kuramoto-logistic-paired": 384,
    "kuramoto-logistic-cell": 96,
}
EXPECTED_KAPPA = {
    "kuramoto-gaussian-paired": [0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6],
    "kuramoto-gaussian-cell": [0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6],
    "kuramoto-logistic-paired": [
        0.7,
        0.8,
        0.875,
        0.925,
        0.975,
        1.0,
        1.025,
        1.075,
        1.15,
        1.2,
        1.4,
        1.5,
    ],
    "kuramoto-logistic-cell": [
        0.7,
        0.8,
        0.875,
        0.925,
        0.975,
        1.0,
        1.025,
        1.075,
        1.15,
        1.2,
        1.4,
        1.5,
    ],
}
PRIMARY_TARGET = "r_full_future"
SENSITIVITY_TARGET = "r_unobserved_future"


def _records(data_dir: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(data_dir.glob("*/*/meta.json")):
        meta = load_json(meta_path)
        params = meta["generator"]["resolved_params"]
        experiment = meta.get("experiment") or {}
        sampling = meta.get("sampling_design") or {}
        control = meta["generator"].get("control") or {}
        if not experiment or not sampling or not control:
            raise RuntimeError(f"{meta_path}: missing frozen experiment/design/control metadata")
        truth_path = meta_path.parent / meta["generator"]["ground_truth"]["path"]
        with np.load(truth_path) as truth:
            means = {}
            for name in (
                "r_full",
                "r_observed",
                "r_unobserved",
                "r_full_future",
                "r_unobserved_future",
            ):
                values = np.asarray(truth[name])
                if values.shape != (1000,):
                    raise RuntimeError(f"{truth_path}: expected {name} shape (1000,), got {values.shape}")
                means[name] = float(np.mean(values))
            critical = float(truth["critical_coupling"])
        class_name = str(meta["mts_class"])
        distribution = str(params["frequency_distribution"])
        role = str(sampling["role"])
        if role not in {"paired-control-path", "independent-cell"}:
            raise RuntimeError(f"{meta_path}: unsupported sampling-design role {role!r}")
        design = "paired" if role == "paired-control-path" else "cell"
        kappa = float(control["reduced_value"])
        if not np.isclose(kappa, float(params["K"]) / critical, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"{meta_path}: stored kappa is inconsistent with K/Kc")
        rows.append(
            {
                "path": meta_path.parent,
                "class_name": class_name,
                "distribution": distribution,
                "design": design,
                "instance": int(meta["instance_index"]),
                "seed": int(meta["generator"]["seed"]),
                "K": float(params["K"]),
                "kappa": kappa,
                "seed_scope": str(sampling["seed_scope"]),
                "seed_group_id": str(sampling["seed_group_id"]),
                "experiment_config_sha256": str(experiment["config_sha256"]),
                "generation_git_commit": str(experiment["git_commit"]),
                "generation_git_dirty": experiment["git_dirty"],
                "config_sha256": str(meta["pyspi"]["config_sha256"]),
                "computation_version": str(meta["pyspi"]["version"]["computation"]),
                "compute_seconds": float(meta["job"]["compute_seconds"]),
                "n_spi_errors": len(meta["pyspi"].get("errors", {})),
                **means,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"no completed datasets under {data_dir}")
    counts = frame.groupby("class_name").size().to_dict()
    if counts != EXPECTED:
        raise RuntimeError(f"incomplete benchmark: found {counts}, expected {EXPECTED}")
    for class_name, expected_grid in EXPECTED_KAPPA.items():
        observed_grid = sorted(
            frame.loc[frame["class_name"] == class_name, "kappa"].round(6).unique()
        )
        if len(observed_grid) != len(expected_grid) or not np.allclose(
            observed_grid, expected_grid, rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(
                f"{class_name}: observed kappa grid {observed_grid}, expected {expected_grid}"
            )
    if frame["config_sha256"].nunique() != 1 or frame["computation_version"].nunique() != 1:
        raise RuntimeError("pyspi configuration/version mismatch across datasets")
    if frame["experiment_config_sha256"].nunique() != 1:
        raise RuntimeError("experiment configuration mismatch across datasets")
    if frame["generation_git_commit"].nunique() != 1 or frame["generation_git_dirty"].ne(False).any():
        raise RuntimeError("generation checkout must have one clean recorded git commit")
    if not (
        frame.loc[frame["design"] == "paired", "seed_scope"].eq("instance").all()
        and frame.loc[frame["design"] == "cell", "seed_scope"].eq("dataset").all()
    ):
        raise RuntimeError("sampling-design and seed-scope metadata are inconsistent")
    frame["kappa_group"] = frame["kappa"].round(6).astype(str)
    frame["split"] = "evaluation"
    frame.loc[
        (frame["distribution"] == "gaussian")
        & (frame["design"] == "paired")
        & (frame["instance"] < 16),
        "split",
    ] = "development"
    return frame.reset_index(drop=True)


def _spi_scalar_matrix(
    paths: list[Path], catalog: list[dict], names: list[str]
) -> tuple[np.ndarray, list[str]]:
    by_name = {str(info["name"]): info for info in catalog}
    summaries = ("mean", "mean_abs", "dispersion", "leading_eigen_fraction")
    labels = [f"{name}::{summary}" for name in names for summary in summaries]
    matrix = np.full((len(paths), len(labels)), np.nan, dtype=np.float64)
    for row, path in enumerate(paths):
        with np.load(path / "spi_mpis.npz") as archive:
            for spi_index, name in enumerate(names):
                raw = np.asarray(archive[name], dtype=np.float64)
                symmetric = 0.5 * (raw + raw.T)
                vector = _edge_vectors(
                    name,
                    raw,
                    bool(by_name[name].get("directed", False)),
                    False,
                )[0][1]
                vector = np.asarray(vector, dtype=np.float64)
                if not np.isfinite(vector).all():
                    continue
                np.fill_diagonal(symmetric, 0.0)
                if not np.isfinite(symmetric).all():
                    continue
                eigenvalues = np.linalg.eigvalsh(symmetric)
                eigen_scale = float(np.sum(np.abs(eigenvalues)))
                values = (
                    float(np.mean(vector)),
                    float(np.mean(np.abs(vector))),
                    float(np.std(vector)),
                    (
                        float(np.max(np.abs(eigenvalues)) / eigen_scale)
                        if eigen_scale > 0.0
                        else float("nan")
                    ),
                )
                start = spi_index * len(summaries)
                matrix[row, start : start + len(summaries)] = values
    return matrix, labels


def _fit_isotonic(feature: np.ndarray, target: np.ndarray, fit: np.ndarray):
    model = IsotonicRegression(
        increasing="auto", out_of_bounds="clip", y_min=0.0, y_max=1.0
    )
    model.fit(feature[fit], target[fit])
    return model, np.asarray(model.predict(feature), dtype=np.float64)


def _metric_block(
    frame: pd.DataFrame,
    target: np.ndarray,
    coordinate: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> dict:
    groups = frame.loc[mask, "kappa_group"].to_numpy()
    coordinate_values = coordinate[mask]
    target_values = target[mask]
    prediction_values = prediction[mask]
    return {
        "n": int(mask.sum()),
        "mae": float(mean_absolute_error(target_values, prediction_values)),
        "rmse": float(mean_squared_error(target_values, prediction_values) ** 0.5),
        "r2": float(r2_score(target_values, prediction_values)),
        "spearman": safe_spearman(coordinate_values, target_values),
        "within_kappa_spearman": safe_spearman(
            residualize_by_group(coordinate_values, groups),
            residualize_by_group(target_values, groups),
        ),
    }


def _reliability_block(
    frame: pd.DataFrame,
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
) -> dict:
    groups = frame.loc[mask, "kappa_group"].to_numpy()
    return {
        "mae": float(mean_absolute_error(first[mask], second[mask])),
        "spearman": safe_spearman(first[mask], second[mask]),
        "within_kappa_spearman": safe_spearman(
            residualize_by_group(first[mask], groups),
            residualize_by_group(second[mask], groups),
        ),
    }


def _json_clean(value):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _ci_excludes_zero(interval) -> bool:
    lower, upper = np.asarray(interval, dtype=np.float64)
    return bool(np.isfinite([lower, upper]).all() and (lower > 0.0 or upper < 0.0))


def _model_payload(prefix: str, model) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_feature_indices": model.feature_indices,
        f"{prefix}_impute_values": model.impute_values,
        f"{prefix}_center": model.center,
        f"{prefix}_component": model.component,
        f"{prefix}_explained_variance_ratio": np.array(model.explained_variance_ratio),
    }


def _sample_clusters(indices: np.ndarray, clusters: np.ndarray, rng) -> np.ndarray:
    labels = np.unique(clusters[indices])
    sampled = rng.choice(labels, size=labels.size, replace=True)
    return np.concatenate([indices[clusters[indices] == label] for label in sampled])


def _conditional_path_gap(
    target: np.ndarray,
    prediction: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    lower = max(float(np.min(target[first])), float(np.min(target[second])))
    upper = min(float(np.max(target[first])), float(np.max(target[second])))
    if not upper > lower:
        return float("nan")
    first = first[(target[first] >= lower) & (target[first] <= upper)]
    second = second[(target[second] >= lower) & (target[second] <= upper)]
    indices = np.concatenate([first, second])
    x = 2.0 * (target[indices] - lower) / (upper - lower) - 1.0
    path = np.concatenate([np.zeros(len(first)), np.ones(len(second))])
    design = np.column_stack(
        [np.ones(len(x)), x, x**2, x**3, path, path * x, path * x**2, path * x**3]
    )
    coefficients = np.linalg.lstsq(design, prediction[indices], rcond=None)[0]
    grid = np.linspace(-1.0, 1.0, 101)
    difference = (
        coefficients[4]
        + coefficients[5] * grid
        + coefficients[6] * grid**2
        + coefficients[7] * grid**3
    )
    return float(np.sqrt(np.mean(difference**2)))


def _joint_path_noise_bootstrap(
    target: np.ndarray,
    prediction: np.ndarray,
    clusters: np.ndarray,
    gaussian_first: np.ndarray,
    gaussian_second: np.ndarray,
    logistic: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jointly resample shared Gaussian masters for path and noise gaps."""
    rng = np.random.default_rng(seed)
    cross = np.empty(n_resamples, dtype=np.float64)
    noise = np.empty(n_resamples, dtype=np.float64)
    for draw in range(n_resamples):
        first_draw = _sample_clusters(gaussian_first, clusters, rng)
        second_draw = _sample_clusters(gaussian_second, clusters, rng)
        gaussian_draw = np.concatenate([first_draw, second_draw])
        logistic_draw = _sample_clusters(logistic, clusters, rng)
        cross[draw] = _conditional_path_gap(
            target, prediction, gaussian_draw, logistic_draw
        )
        noise[draw] = _conditional_path_gap(
            target, prediction, first_draw, second_draw
        )
    return cross, noise, cross - noise


def _curve_rmse(
    values: np.ndarray,
    groups: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    shared = sorted(set(groups[first]) & set(groups[second]))
    differences = []
    for group in shared:
        differences.append(
            float(np.mean(values[first[groups[first] == group]]))
            - float(np.mean(values[second[groups[second] == group]]))
        )
    return float(np.sqrt(np.mean(np.square(differences))))


def _stratified_row_sample(
    indices: np.ndarray, groups: np.ndarray, rng
) -> np.ndarray:
    sampled = []
    for group in np.unique(groups[indices]):
        cell = indices[groups[indices] == group]
        sampled.append(rng.choice(cell, size=len(cell), replace=True))
    return np.concatenate(sampled)


def _paired_cell_bootstrap(
    values: np.ndarray,
    groups: np.ndarray,
    clusters: np.ndarray,
    paired: np.ndarray,
    cell: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for draw in range(n_resamples):
        paired_draw = _sample_clusters(paired, clusters, rng)
        cell_draw = _stratified_row_sample(cell, groups, rng)
        draws[draw] = _curve_rmse(
            values, groups, paired_draw, cell_draw
        )
    return draws


def _missingness_gate(
    missing: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    evaluation: np.ndarray,
    *,
    max_allowed: float,
    p95_allowed: float,
    association_allowed: float,
) -> dict:
    values = missing[evaluation]
    target_values = target[evaluation]
    group_values = groups[evaluation]
    overall = safe_spearman(values, target_values)
    within = safe_spearman(
        residualize_by_group(values, group_values),
        residualize_by_group(target_values, group_values),
    )
    result = {
        "max_allowed": float(max_allowed),
        "p95_allowed": float(p95_allowed),
        "absolute_target_spearman_allowed": float(association_allowed),
        "observed_max": float(np.max(values)),
        "observed_p95": float(np.quantile(values, 0.95)),
        "target_spearman": overall,
        "within_kappa_target_spearman": within,
    }
    result["passed"] = bool(
        result["observed_max"] <= result["max_allowed"]
        and result["observed_p95"] <= result["p95_allowed"]
        and (not np.isfinite(overall) or abs(overall) <= association_allowed)
        and (not np.isfinite(within) or abs(within) <= association_allowed)
    )
    return result


def _truth_result_payload(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Canonical and sensitivity truth arrays with unambiguous labels."""
    return {
        "r_full_future": frame["r_full_future"].to_numpy(),
        "r_unobserved_future": frame["r_unobserved_future"].to_numpy(),
        "r_full": frame["r_full"].to_numpy(),
        "r_unobserved": frame["r_unobserved"].to_numpy(),
        "r_observed": frame["r_observed"].to_numpy(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_order_benchmark",
    )
    parser.add_argument("--variance-threshold", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--max-evaluation-missing-fraction", type=float, default=0.05)
    parser.add_argument("--p95-evaluation-missing-fraction", type=float, default=0.01)
    parser.add_argument("--max-missing-target-spearman", type=float, default=0.20)
    args = parser.parse_args()

    frame = _records(args.data_dir)
    paths = frame["path"].tolist()
    development = frame["split"].eq("development").to_numpy()
    target = frame[PRIMARY_TARGET].to_numpy()
    sensitivity_target = frame[SENSITIVITY_TARGET].to_numpy()

    catalog = validate_spi_catalogs(paths)
    development_paths = frame.loc[development, "path"].tolist()
    stable, validity = stable_spi_names(
        development_paths, catalog, min_valid_fraction=1.0
    )
    meta, pairs = build_meta_feature_matrix(paths, catalog, stable, metric="pearson")
    spi_model = fit_frozen_pc1(
        meta[development], variance_threshold=args.variance_threshold
    )
    q_spi, spi_missing = spi_model.transform(meta)

    phase_names = explicit_phase_spi_names(catalog)
    no_phase_columns = np.array(
        [left not in phase_names and right not in phase_names for left, right in pairs]
    )
    no_phase_meta = meta[:, no_phase_columns]
    no_phase_model = fit_frozen_pc1(
        no_phase_meta[development], variance_threshold=args.variance_threshold
    )
    q_no_phase, no_phase_missing = no_phase_model.transform(no_phase_meta)

    complete_stable_columns = np.isfinite(meta).all(axis=0)
    complete_stable_meta = meta[:, complete_stable_columns]
    complete_stable_model = fit_frozen_pc1(
        complete_stable_meta[development],
        variance_threshold=args.variance_threshold,
    )
    q_complete_stable, complete_stable_missing = complete_stable_model.transform(
        complete_stable_meta
    )

    input_rows = [input_only_features(np.load(path / "timeseries.npy")) for path in paths]
    correlation_vectors = np.vstack([row["correlation_vector"] for row in input_rows])
    correlation_model = fit_frozen_pc1(
        correlation_vectors[development], variance_threshold=1e-6
    )
    q_correlation, correlation_missing = correlation_model.transform(correlation_vectors)
    scalar_coordinates = {
        "spi_spi_pc1": q_spi,
        "spi_spi_pc1_no_phase": q_no_phase,
        "spi_spi_pc1_complete_stable_sensitivity": q_complete_stable,
        "input_correlation_pc1": q_correlation,
        "mean_abs_correlation": np.array(
            [row["mean_abs_correlation"] for row in input_rows]
        ),
        "covariance_leading_fraction": np.array(
            [row["covariance_leading_fraction"] for row in input_rows]
        ),
        "analytic_phase_coherence": np.array(
            [row["analytic_phase_coherence"] for row in input_rows]
        ),
        "mean_temporal_spectral_entropy": np.array(
            [row["mean_temporal_spectral_entropy"] for row in input_rows]
        ),
        "pooled_std": np.array([row["pooled_std"] for row in input_rows]),
        "control_kappa": frame["kappa"].to_numpy(),
        "phase_oracle_r_m": frame["r_observed"].to_numpy(),
    }

    spi_scalars, spi_scalar_labels = _spi_scalar_matrix(paths, catalog, stable)
    development_medians = np.nanmedian(spi_scalars[development], axis=0)
    invalid = ~np.isfinite(spi_scalars)
    if np.any(invalid):
        rows, columns = np.where(invalid)
        spi_scalars[rows, columns] = development_medians[columns]
    target_residual = residualize_by_group(
        target[development], frame.loc[development, "kappa_group"].to_numpy()
    )
    candidate_scores = np.array(
        [
            abs(
                safe_spearman(
                    residualize_by_group(
                        spi_scalars[development, column],
                        frame.loc[development, "kappa_group"].to_numpy(),
                    ),
                    target_residual,
                )
            )
            for column in range(spi_scalars.shape[1])
        ]
    )
    candidate_scores = np.nan_to_num(candidate_scores, nan=-np.inf)
    best_spi_index = int(np.argmax(candidate_scores))
    scalar_coordinates["train_selected_individual_spi_summary"] = spi_scalars[
        :, best_spi_index
    ]

    calibrators = {}
    predictions = {}
    for name, coordinate in scalar_coordinates.items():
        calibrators[name], predictions[name] = _fit_isotonic(
            np.asarray(coordinate), target, development
        )
    predictions["phase_oracle_r_m_direct"] = scalar_coordinates["phase_oracle_r_m"].copy()

    evaluation_sets = {
        "development": development,
        "gaussian_holdout": (
            (frame["distribution"] == "gaussian")
            & (frame["design"] == "paired")
            & (frame["instance"] >= 16)
        ).to_numpy(),
        "logistic_paired": (
            (frame["distribution"] == "logistic") & (frame["design"] == "paired")
        ).to_numpy(),
        "gaussian_independent_cell": frame["class_name"]
        .eq("kuramoto-gaussian-cell")
        .to_numpy(),
        "logistic_independent_cell": frame["class_name"]
        .eq("kuramoto-logistic-cell")
        .to_numpy(),
    }
    target_reliability = {}
    for set_name, mask in evaluation_sets.items():
        target_reliability[set_name] = {
            "current_full_vs_future_full": _reliability_block(
                frame,
                frame["r_full"].to_numpy(),
                frame["r_full_future"].to_numpy(),
                mask,
            ),
            "current_complement_vs_future_complement": _reliability_block(
                frame,
                frame["r_unobserved"].to_numpy(),
                frame["r_unobserved_future"].to_numpy(),
                mask,
            ),
            "future_full_vs_future_complement": _reliability_block(
                frame,
                frame["r_full_future"].to_numpy(),
                frame["r_unobserved_future"].to_numpy(),
                mask,
            ),
        }

    evaluation = ~development
    kappa_groups = frame["kappa_group"].to_numpy()
    missingness_gates = {
        name: _missingness_gate(
            values,
            target,
            kappa_groups,
            evaluation,
            max_allowed=args.max_evaluation_missing_fraction,
            p95_allowed=args.p95_evaluation_missing_fraction,
            association_allowed=args.max_missing_target_spearman,
        )
        for name, values in {
            "spi_spi_pc1": spi_missing,
            "spi_spi_pc1_no_phase": no_phase_missing,
        }.items()
    }
    missingness_by_path_and_kappa = {}
    for representation, missing in {
        "spi_spi_pc1": spi_missing,
        "spi_spi_pc1_no_phase": no_phase_missing,
    }.items():
        missingness_by_path_and_kappa[representation] = {}
        for (class_name, kappa), indices in frame.loc[evaluation].groupby(
            ["class_name", "kappa_group"]
        ).groups.items():
            values = missing[np.asarray(list(indices), dtype=int)]
            missingness_by_path_and_kappa[representation][f"{class_name}:{kappa}"] = {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
            }
    claim_eligibility = {
        representation: {
            "eligible": bool(gate["passed"]),
            "reason": (
                "predeclared evaluation missingness gate passed"
                if gate["passed"]
                else "ineligible: predeclared evaluation missingness gate failed"
            ),
        }
        for representation, gate in missingness_gates.items()
    }

    metrics = {}
    for set_name, mask in evaluation_sets.items():
        metrics[set_name] = {}
        for method, prediction in predictions.items():
            coordinate_name = (
                "phase_oracle_r_m" if method == "phase_oracle_r_m_direct" else method
            )
            metrics[set_name][method] = _metric_block(
                frame,
                target,
                scalar_coordinates[coordinate_name],
                prediction,
                mask,
            )

    sensitivity_predictions = {}
    sensitivity_metrics = {}
    for method in (
        "spi_spi_pc1",
        "spi_spi_pc1_no_phase",
        "spi_spi_pc1_complete_stable_sensitivity",
        "control_kappa",
        "analytic_phase_coherence",
    ):
        _, sensitivity_predictions[method] = _fit_isotonic(
            scalar_coordinates[method], sensitivity_target, development
        )
    for set_name, mask in evaluation_sets.items():
        sensitivity_metrics[set_name] = {
            method: _metric_block(
                frame,
                sensitivity_target,
                scalar_coordinates[method],
                prediction,
                mask,
            )
            for method, prediction in sensitivity_predictions.items()
        }

    bootstrap = {}
    association_bootstrap = {}
    calibration_bootstrap = {}
    sensitivity_association_bootstrap = {}
    for set_name in ("gaussian_holdout", "logistic_paired"):
        mask = evaluation_sets[set_name]
        clusters = frame.loc[mask, "instance"].to_numpy()
        bootstrap[set_name] = {}
        association_bootstrap[set_name] = {}
        calibration_bootstrap[set_name] = {}
        sensitivity_association_bootstrap[set_name] = {}
        for baseline in (
            "control_kappa",
            "analytic_phase_coherence",
            "input_correlation_pc1",
            "train_selected_individual_spi_summary",
        ):
            draws = clustered_bootstrap_difference(
                target[mask],
                predictions["spi_spi_pc1"][mask],
                predictions[baseline][mask],
                clusters,
                n_resamples=args.bootstrap_resamples,
                seed=8137 + len(bootstrap[set_name]),
            )
            bootstrap[set_name][f"spi_spi_pc1_minus_{baseline}"] = {
                "mean": float(np.mean(draws)),
                "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
                "probability_spi_spi_lower_mae": float(np.mean(draws < 0.0)),
            }
        for method in (
            "spi_spi_pc1",
            "spi_spi_pc1_no_phase",
            "spi_spi_pc1_complete_stable_sensitivity",
            "analytic_phase_coherence",
            "input_correlation_pc1",
            "train_selected_individual_spi_summary",
        ):
            overall, within = clustered_bootstrap_spearman(
                scalar_coordinates[method][mask],
                target[mask],
                frame.loc[mask, "kappa_group"].to_numpy(),
                clusters,
                n_resamples=args.bootstrap_resamples,
                seed=9173 + len(association_bootstrap[set_name]),
            )
            association_bootstrap[set_name][method] = {
                "overall_ci95": np.nanquantile(overall, [0.025, 0.975]).tolist(),
                "within_kappa_ci95": np.nanquantile(within, [0.025, 0.975]).tolist(),
            }
        for method in (
            "spi_spi_pc1",
            "spi_spi_pc1_no_phase",
            "spi_spi_pc1_complete_stable_sensitivity",
            "control_kappa",
            "analytic_phase_coherence",
        ):
            mae_draws = clustered_bootstrap_mae(
                target[mask],
                predictions[method][mask],
                clusters,
                n_resamples=args.bootstrap_resamples,
                seed=10103 + len(calibration_bootstrap[set_name]),
            )
            calibration_bootstrap[set_name][method] = {
                "mae_ci95": np.quantile(mae_draws, [0.025, 0.975]).tolist()
            }
            overall, within = clustered_bootstrap_spearman(
                scalar_coordinates[method][mask],
                sensitivity_target[mask],
                frame.loc[mask, "kappa_group"].to_numpy(),
                clusters,
                n_resamples=args.bootstrap_resamples,
                seed=11117 + len(sensitivity_association_bootstrap[set_name]),
            )
            sensitivity_association_bootstrap[set_name][method] = {
                "overall_ci95": np.nanquantile(overall, [0.025, 0.975]).tolist(),
                "within_kappa_ci95": np.nanquantile(within, [0.025, 0.975]).tolist(),
            }

    gaussian_mask = evaluation_sets["gaussian_holdout"]
    logistic_mask = evaluation_sets["logistic_paired"]
    gaussian_indices = np.flatnonzero(gaussian_mask)
    logistic_indices = np.flatnonzero(logistic_mask)
    global_clusters = frame["instance"].to_numpy()
    gaussian_noise_first = gaussian_indices[frame.loc[gaussian_indices, "instance"].to_numpy() < 24]
    gaussian_noise_second = gaussian_indices[frame.loc[gaussian_indices, "instance"].to_numpy() >= 24]
    conditional_path_effect = {}
    for method in ("spi_spi_pc1", "spi_spi_pc1_no_phase"):
        estimate = predictions[method]
        cross_draws, noise_draws, difference_draws = _joint_path_noise_bootstrap(
            target,
            estimate,
            global_clusters,
            gaussian_noise_first,
            gaussian_noise_second,
            logistic_indices,
            n_resamples=args.bootstrap_resamples,
            seed=12143 + len(conditional_path_effect),
        )
        cross_estimate = _conditional_path_gap(
            target, estimate, gaussian_indices, logistic_indices
        )
        noise_estimate = _conditional_path_gap(
            target, estimate, gaussian_noise_first, gaussian_noise_second
        )
        conditional_path_effect[method] = {
            "cross_path_rms_predicted_r_gap": cross_estimate,
            "cross_path_ci95": np.nanquantile(
                cross_draws, [0.025, 0.975]
            ).tolist(),
            "gaussian_split_noise_floor": noise_estimate,
            "gaussian_split_noise_floor_ci95": np.nanquantile(
                noise_draws, [0.025, 0.975]
            ).tolist(),
            "cross_minus_noise_descriptive": cross_estimate - noise_estimate,
            "cross_minus_noise_joint_ci95": np.nanquantile(
                difference_draws, [0.025, 0.975]
            ).tolist(),
            "difference_interpretation": (
                "descriptive only; no equivalence margin was predeclared"
            ),
        }

    shared_kappa = sorted(
        set(frame.loc[gaussian_mask, "kappa_group"])
        & set(frame.loc[logistic_mask, "kappa_group"])
    )
    group_values = frame["kappa_group"].to_numpy()
    matched_kappa = []
    for kappa in shared_kappa:
        g = gaussian_indices[group_values[gaussian_indices] == kappa]
        l = logistic_indices[group_values[logistic_indices] == kappa]
        actual_delta = float(target[l].mean() - target[g].mean())
        predicted_delta = float(
            predictions["spi_spi_pc1"][l].mean()
            - predictions["spi_spi_pc1"][g].mean()
        )
        rng = np.random.default_rng(14159 + len(matched_kappa))
        errors = np.empty(args.bootstrap_resamples, dtype=np.float64)
        for draw in range(args.bootstrap_resamples):
            g_draw_all = _sample_clusters(gaussian_indices, global_clusters, rng)
            l_draw_all = _sample_clusters(logistic_indices, global_clusters, rng)
            g_draw = g_draw_all[group_values[g_draw_all] == kappa]
            l_draw = l_draw_all[group_values[l_draw_all] == kappa]
            draw_actual = float(target[l_draw].mean() - target[g_draw].mean())
            draw_predicted = float(
                predictions["spi_spi_pc1"][l_draw].mean()
                - predictions["spi_spi_pc1"][g_draw].mean()
            )
            errors[draw] = draw_predicted - draw_actual
        matched_kappa.append(
            {
                "kappa": float(kappa),
                "actual_delta_r_logistic_minus_gaussian": actual_delta,
                "predicted_delta_r_logistic_minus_gaussian": predicted_delta,
                "prediction_error": predicted_delta - actual_delta,
                "prediction_error_ci95": np.quantile(
                    errors, [0.025, 0.975]
                ).tolist(),
            }
        )

    paired_cell_agreement = {}
    for distribution in ("gaussian", "logistic"):
        paired = np.flatnonzero(evaluation_sets[f"{distribution}_holdout"] if distribution == "gaussian" else logistic_mask)
        cell = np.flatnonzero(evaluation_sets[f"{distribution}_independent_cell"])
        paired_cell_agreement[distribution] = {}
        for name, values in (
            ("canonical_target", target),
            ("spi_spi_predicted_target", predictions["spi_spi_pc1"]),
        ):
            draws = _paired_cell_bootstrap(
                values,
                group_values,
                global_clusters,
                paired,
                cell,
                n_resamples=args.bootstrap_resamples,
                seed=15173 + len(paired_cell_agreement[distribution]),
            )
            paired_cell_agreement[distribution][name] = {
                "curve_rmse": _curve_rmse(
                    values, group_values, paired, cell
                ),
                "curve_rmse_ci95": np.quantile(
                    draws, [0.025, 0.975]
                ).tolist(),
            }

    analysis_git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    held_out_paths = ("gaussian_holdout", "logistic_paired")
    primary_overall = all(
        _ci_excludes_zero(
            association_bootstrap[path]["spi_spi_pc1"]["overall_ci95"]
        )
        for path in held_out_paths
    )
    primary_within = all(
        _ci_excludes_zero(
            association_bootstrap[path]["spi_spi_pc1"]["within_kappa_ci95"]
        )
        for path in held_out_paths
    )
    complement_within = all(
        _ci_excludes_zero(
            sensitivity_association_bootstrap[path]["spi_spi_pc1"][
                "within_kappa_ci95"
            ]
        )
        for path in held_out_paths
    )
    no_phase_association = all(
        _ci_excludes_zero(
            association_bootstrap[path]["spi_spi_pc1_no_phase"]["overall_ci95"]
        )
        and _ci_excludes_zero(
            association_bootstrap[path]["spi_spi_pc1_no_phase"][
                "within_kappa_ci95"
            ]
        )
        for path in held_out_paths
    )
    transfer_advantage_ci = bootstrap["logistic_paired"][
        "spi_spi_pc1_minus_control_kappa"
    ]["ci95"]
    level_1 = bool(missingness_gates["spi_spi_pc1"]["passed"] and primary_overall)
    level_2 = bool(level_1 and primary_within and complement_within)
    level_3 = bool(
        level_2
        and transfer_advantage_ci[1] < 0.0
        and missingness_gates["spi_spi_pc1_no_phase"]["passed"]
        and no_phase_association
    )
    claim_decision = {
        "level_1_transition_coordinate": level_1,
        "level_2_realization_level_order_coordinate": level_2,
        "level_3_cross_path_order_parameter_inference": level_3,
        "maximum_supported_level": 3 if level_3 else 2 if level_2 else 1 if level_1 else 0,
        "criteria": {
            "level_1": "primary missingness gate passes and held-out overall association CIs exclude zero on both paths",
            "level_2": "level 1 plus within-kappa full-system and hidden-complement association CIs exclude zero on both paths",
            "level_3": "level 2 plus Gaussian-trained SPI-SPI calibration beats kappa on logistic data (MAE-difference CI < 0) and the no-phase ablation passes missingness and association criteria",
        },
        "logistic_spi_spi_minus_kappa_mae_ci95": transfer_advantage_ci,
    }
    summary = {
        "datasets": int(len(frame)),
        "development_datasets": int(development.sum()),
        "primary_target": PRIMARY_TARGET,
        "anti_self_inclusion_sensitivity_target": SENSITIVITY_TARGET,
        "analysis_contract": {
            "analysis_git_commit": analysis_git_commit,
            "development_split": (
                "gaussian paired masters with instance 0..15; every kappa for a "
                "master remains in one split"
            ),
            "evaluation_split": (
                "gaussian paired masters 16..31; all logistic paired masters; "
                "independent-cell classes used only for pairing sensitivity"
            ),
            "variance_threshold": float(args.variance_threshold),
            "bootstrap_resamples": int(args.bootstrap_resamples),
            "missingness_gate": {
                "max": float(args.max_evaluation_missing_fraction),
                "p95": float(args.p95_evaluation_missing_fraction),
                "absolute_target_spearman": float(args.max_missing_target_spearman),
            },
            "kappa_grids": EXPECTED_KAPPA,
            "representation": "Pearson SPI-SPI meta-features followed by development-fitted PC1",
            "calibration": "development-fitted isotonic regression; supervised and reported separately",
        },
        "experiment_config_sha256": frame["experiment_config_sha256"].iloc[0],
        "generation_git_commit": frame["generation_git_commit"].iloc[0],
        "pyspi_config_sha256": frame["config_sha256"].iloc[0],
        "pyspi_computation_version": frame["computation_version"].iloc[0],
        "stable_spis": len(stable),
        "total_spis": len(catalog),
        "minimum_development_spi_validity": float(min(validity.values())),
        "explicit_phase_spis_removed": len(set(stable) & phase_names),
        "meta_features": len(pairs),
        "retained_meta_features": int(spi_model.feature_indices.size),
        "pc1_explained_variance": spi_model.explained_variance_ratio,
        "no_phase_pc1_explained_variance": no_phase_model.explained_variance_ratio,
        "max_spi_spi_missing_fraction": float(np.max(spi_missing)),
        "max_no_phase_missing_fraction": float(np.max(no_phase_missing)),
        "complete_stable_meta_features": int(complete_stable_columns.sum()),
        "max_complete_stable_missing_fraction": float(
            np.max(complete_stable_missing)
        ),
        "max_input_correlation_missing_fraction": float(np.max(correlation_missing)),
        "best_individual_spi_summary": spi_scalar_labels[best_spi_index],
        "best_individual_spi_summary_development_score": float(
            candidate_scores[best_spi_index]
        ),
        "runtime_seconds": {
            "median": float(frame["compute_seconds"].median()),
            "p95": float(frame["compute_seconds"].quantile(0.95)),
            "max": float(frame["compute_seconds"].max()),
        },
        "target_reliability": target_reliability,
        "missingness_gates": missingness_gates,
        "claim_eligibility": claim_eligibility,
        "missingness_by_path_and_kappa": missingness_by_path_and_kappa,
        "metrics": metrics,
        "anti_self_inclusion_sensitivity_metrics": sensitivity_metrics,
        "cluster_bootstrap_mae_differences": bootstrap,
        "cluster_bootstrap_calibration_mae": calibration_bootstrap,
        "cluster_bootstrap_associations": association_bootstrap,
        "anti_self_inclusion_cluster_bootstrap_associations": (
            sensitivity_association_bootstrap
        ),
        "conditional_cross_path_effect": conditional_path_effect,
        "matched_kappa_cross_path": matched_kappa,
        "paired_vs_independent_cell_agreement": paired_cell_agreement,
        "claim_decision": claim_decision,
    }

    args.data_dir.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "class_name": frame["class_name"].to_numpy(dtype=str),
        "distribution": frame["distribution"].to_numpy(dtype=str),
        "design": frame["design"].to_numpy(dtype=str),
        "split": frame["split"].to_numpy(dtype=str),
        "instance": frame["instance"].to_numpy(dtype=np.int32),
        "seed": frame["seed"].to_numpy(dtype=np.int64),
        "seed_group_id": frame["seed_group_id"].to_numpy(dtype=str),
        "kappa": frame["kappa"].to_numpy(),
        **_truth_result_payload(frame),
        "spi_spi_missing_fraction": spi_missing,
        "no_phase_missing_fraction": no_phase_missing,
        "complete_stable_missing_fraction": complete_stable_missing,
        **{f"coordinate_{name}": values for name, values in scalar_coordinates.items()},
        **{f"prediction_{name}": values for name, values in predictions.items()},
        **{
            f"sensitivity_prediction_{name}": values
            for name, values in sensitivity_predictions.items()
        },
    }
    np.savez_compressed(args.data_dir / "benchmark_results.npz", **result_payload)

    stable_index = {name: index for index, name in enumerate(stable)}
    pair_left = np.array([stable_index[left] for left, _ in pairs], dtype=np.int32)
    pair_right = np.array([stable_index[right] for _, right in pairs], dtype=np.int32)
    model_payload = {
        "stable_spis": np.asarray(stable, dtype=str),
        "pair_left": pair_left,
        "pair_right": pair_right,
        "no_phase_columns": no_phase_columns,
        "complete_stable_columns": complete_stable_columns,
        "spi_scalar_labels": np.asarray(spi_scalar_labels, dtype=str),
        "best_spi_summary_index": np.array(best_spi_index),
        **_model_payload("spi_spi", spi_model),
        **_model_payload("no_phase", no_phase_model),
        **_model_payload("complete_stable", complete_stable_model),
        **_model_payload("input_correlation", correlation_model),
    }
    for name, calibrator in calibrators.items():
        model_payload[f"calibrator_{name}_x"] = calibrator.X_thresholds_
        model_payload[f"calibrator_{name}_y"] = calibrator.y_thresholds_
    np.savez_compressed(args.data_dir / "benchmark_model.npz", **model_payload)
    (args.data_dir / "stable_spis.txt").write_text("\n".join(stable) + "\n", encoding="utf-8")
    clean_summary = _json_clean(summary)
    (args.data_dir / "benchmark_summary.json").write_text(
        json.dumps(clean_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(clean_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
