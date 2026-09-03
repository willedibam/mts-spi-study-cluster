#!/usr/bin/env python3
"""Fit and evaluate a frozen local Desai--Zwanzig SPI--SPI coordinate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_stuart_landau_development import (  # noqa: E402
    _anchor_components,
    _load_artifact,
    _resolve_dataset_path,
)
from scripts.analyze_stuart_landau_confirmation import _steepest_interval  # noqa: E402
from src.order_parameter_analysis import (  # noqa: E402
    clustered_bootstrap_mae,
    clustered_bootstrap_spearman,
    input_only_features,
    residualize_by_group,
    safe_spearman,
)
from src.spi_spi_analysis import fit_feature_transform  # noqa: E402
from src.utils import load_json, load_yaml  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_frame(payload: dict[str, np.ndarray], data_root: Path) -> pd.DataFrame:
    rows = []
    for raw in np.asarray(payload["dataset_paths"], dtype=object):
        path = _resolve_dataset_path(raw, data_root)
        meta = load_json(path / "meta.json")
        params = meta["generator"]["resolved_params"]
        class_name = str(meta["mts_class"])
        labels = {str(value) for value in meta.get("labels", [])}
        if "mean-field-observation" in class_name:
            arm = "mean_field"
        elif "full-small-system" in class_name:
            arm = "finite_N32"
        else:
            raise ValueError(f"cannot infer Desai--Zwanzig arm for {path}")
        rows.append(
            {
                "path": path,
                "class_name": class_name,
                "arm": arm,
                "observation": (
                    "full" if "full-observation" in labels else "partial"
                ),
                "M": int(meta["M"]),
                "T": int(meta["T"]),
                "instance": int(meta["instance_index"]),
                "sigma": float(params["sigma"]),
                "N_full": int(params["N_full"]),
                "alpha": float(params["alpha"]),
                "theta": float(params["theta"]),
                "sigma_m": float(params["sigma_m"]),
                "nu": float(params["nu"]),
                "integration_scheme": str(params["integration_scheme"]),
                "experiment_commit": str(meta["experiment"]["git_commit"]),
                "experiment_config_sha256": str(
                    meta["experiment"]["config_sha256"]
                ),
                "pyspi_config_sha256": str(meta["pyspi"]["config_sha256"]),
                "pyspi_version": json.dumps(meta["pyspi"]["version"], sort_keys=True),
                "experiment_dirty": bool(meta["experiment"]["git_dirty"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["sigma_group"] = frame["sigma"].map(lambda value: f"{value:.6g}")
    return frame


def _load_targets(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        with np.load(path / "ground_truth.npz", allow_pickle=False) as truth:
            rows.append(
                {
                    "Q_mean_abs": float(truth["q_mean_abs"]),
                    "Q_mean_signed": float(truth["q_mean_signed"]),
                    "Q_rms": float(truth["q_mean_rms"]),
                    "Q_block_range": float(np.ptp(truth["q_mean_abs_blocks"])),
                }
            )
    return pd.DataFrame(rows)


def _input_baselines(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        values = np.load(path / "timeseries.npy").astype(np.float64, copy=False)
        features = input_only_features(values)
        rows.append(
            {
                "observed_mean_abs": float(np.mean(np.abs(np.mean(values, axis=1)))),
                "mean_abs_correlation": float(features["mean_abs_correlation"]),
                "covariance_leading_fraction": float(
                    features["covariance_leading_fraction"]
                ),
                "temporal_spectral_entropy": float(
                    features["mean_temporal_spectral_entropy"]
                ),
                "pooled_std": float(features["pooled_std"]),
            }
        )
    return pd.DataFrame(rows)


def _association(frame: pd.DataFrame, mask: np.ndarray, value: str) -> dict[str, object]:
    selected = frame.loc[mask]
    overall, within = clustered_bootstrap_spearman(
        selected[value],
        selected["Q_mean_abs"],
        selected["sigma_group"],
        selected["instance"],
        n_resamples=2000,
        seed=29417,
    )
    q = selected[value].to_numpy()
    target = selected["Q_mean_abs"].to_numpy()
    groups = selected["sigma_group"].to_numpy()
    cell = selected.groupby("sigma")[[value, "Q_mean_abs"]].mean()
    return {
        "overall_spearman": safe_spearman(q, target),
        "overall_ci95": np.quantile(overall, [0.025, 0.975]).tolist(),
        "within_sigma_spearman": safe_spearman(
            residualize_by_group(q, groups),
            residualize_by_group(target, groups),
        ),
        "within_sigma_ci95": np.quantile(within, [0.025, 0.975]).tolist(),
        "control_mean_spearman": safe_spearman(cell[value], cell["Q_mean_abs"]),
    }


def _normalized_maximum_adjacent_change(curve: pd.Series) -> dict[str, object]:
    curve = curve.sort_index()
    values = curve.to_numpy(dtype=float)
    controls = curve.index.to_numpy(dtype=float)
    value_range = float(np.ptp(values))
    changes = np.abs(np.diff(values)) / value_range if value_range > 0 else np.zeros(len(values) - 1)
    index = int(np.argmax(changes))
    return {
        "fraction_of_total_range": float(changes[index]),
        "interval": [float(controls[index]), float(controls[index + 1])],
    }


def _pca_stability(
    transformed: np.ndarray,
    frame: pd.DataFrame,
    fit_mask: np.ndarray,
    reference: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    for instance in sorted(frame.loc[fit_mask, "instance"].unique()):
        mask = fit_mask & frame["instance"].ne(instance).to_numpy()
        model = PCA(
            n_components=1,
            svd_solver="randomized",
            iterated_power=7,
            random_state=seed + int(instance),
        ).fit(transformed[mask])
        component = _anchor_components(model.components_)[0]
        rows.append(
            {
                "left_out_instance": int(instance),
                "loading_cosine_absolute": float(abs(np.dot(component, reference))),
                "explained_variance_ratio": float(model.explained_variance_ratio_[0]),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=86411)
    args = parser.parse_args()

    contract = load_yaml(args.analysis_contract)
    representation = contract["representation"]
    design = contract["expected_design"]
    eligibility_contract = contract["eligibility"]
    expected_sigmas = {float(value) for value in design["sigmas"]}
    fit_instances = {int(value) for value in representation["fit_instances"]}
    confirmation_instances = {
        int(value) for value in representation["confirmation_instances"]
    }

    payload = _load_artifact(args.features)
    values = np.asarray(payload["X"], dtype=np.float32)
    frame = _metadata_frame(payload, args.data_root)
    fit_mask = (
        frame["arm"].eq(representation["fit_arm"])
        & frame["instance"].isin(fit_instances)
    ).to_numpy()
    expected_rows = len(design["arms"]) * len(expected_sigmas) * int(design["instances"])
    design_valid = (
        len(frame) == expected_rows
        and set(frame["arm"]) == set(design["arms"])
        and set(frame["M"]) == {int(design["M"])}
        and set(frame["T"]) == {int(design["T"])}
        and set(frame["instance"]) == set(range(int(design["instances"])))
        and set(frame["sigma"]) == expected_sigmas
        and set(frame["alpha"]) == {1.0}
        and set(frame["theta"]) == {4.0}
        and set(frame["sigma_m"]) == {0.8}
        and set(frame["nu"]) == {0.5}
        and set(frame["integration_scheme"]) == {"milstein"}
        and set(frame.loc[frame["arm"].eq("mean_field"), "N_full"]) == {12000}
        and set(frame.loc[frame["arm"].eq("finite_N32"), "N_full"]) == {32}
        and set(frame.loc[frame["arm"].eq("mean_field"), "observation"]) == {"partial"}
        and set(frame.loc[frame["arm"].eq("finite_N32"), "observation"]) == {"full"}
        and int(fit_mask.sum()) == len(expected_sigmas) * len(fit_instances)
    )
    provenance_valid = all(
        frame[column].nunique() == 1
        for column in (
            "experiment_commit",
            "experiment_config_sha256",
            "pyspi_config_sha256",
            "pyspi_version",
            "experiment_dirty",
        )
    ) and not bool(frame["experiment_dirty"].iloc[0])

    transform = fit_feature_transform(
        values[fit_mask],
        np.asarray(payload["feature_block"], dtype=str),
        minimum_valid_fraction=float(representation["minimum_valid_fraction"]),
        variance_threshold=float(representation["variance_threshold"]),
        block_balanced=False,
    )
    transformed = transform.transform(values)
    component_count = min(10, transformed[fit_mask].shape[0] - 1, transformed.shape[1])
    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        iterated_power=7,
        random_state=args.seed,
    ).fit(transformed[fit_mask])
    components = _anchor_components(pca.components_)
    raw_scores = transformed @ components.T
    q_center = float(np.mean(raw_scores[fit_mask, 0]))
    q_scale = float(np.std(raw_scores[fit_mask, 0]))
    if not q_scale > 0.0:
        raise RuntimeError("target-blind PC1 is constant on the fit rows")
    frame["q"] = (raw_scores[:, 0] - q_center) / q_scale
    selected_missingness = np.mean(
        ~np.isfinite(values[:, transform.keep_indices]), axis=1
    )
    frame["selected_missingness"] = selected_missingness
    stability = _pca_stability(
        transformed, frame, fit_mask, components[0], args.seed + 100
    )
    minimum_cosine = min(row["loading_cosine_absolute"] for row in stability)

    threshold = float(
        eligibility_contract["maximum_selected_feature_missingness_per_row"]
    )
    retained = selected_missingness <= threshold
    confirmation_mask = frame["instance"].isin(confirmation_instances).to_numpy()
    retained_counts = (
        frame.assign(_retained=retained & confirmation_mask)
        .groupby(["arm", "sigma"])["_retained"]
        .sum()
    )
    excluded_fraction = float(np.mean(~retained))
    missingness_gate = (
        bool(retained[fit_mask].all())
        and excluded_fraction
        <= float(eligibility_contract["maximum_excluded_fraction"])
        and int(retained_counts.min())
        >= int(
            eligibility_contract[
                "minimum_retained_confirmation_instances_per_control_arm"
            ]
        )
    )
    gates = {
        "complete_expected_design": bool(design_valid),
        "homogeneous_clean_provenance": bool(provenance_valid),
        "frozen_schema_and_full_spi_order": bool(
            np.asarray(payload["spi_order"]).size == 289
            and values.shape[1] == 41616
        ),
        "selected_missingness_policy_passes": bool(missingness_gate),
        "all_retained_coordinates_finite": bool(
            np.isfinite(frame.loc[retained, "q"]).all()
        ),
        "pc1_explained_variance_gate": bool(
            pca.explained_variance_ratio_[0]
            >= float(eligibility_contract["minimum_pc1_explained_variance"])
        ),
        "leave_instance_loading_stability_gate": bool(
            minimum_cosine
            >= float(eligibility_contract["minimum_leave_instance_loading_cosine"])
        ),
    }
    eligibility = {
        "status": "eligible" if all(gates.values()) else "ineligible",
        "outcomes_read": False,
        "rows": int(len(frame)),
        "fit_rows": int(fit_mask.sum()),
        "retained_rows": int(retained.sum()),
        "excluded_rows": int((~retained).sum()),
        "excluded_fraction": excluded_fraction,
        "minimum_retained_confirmation_per_control_arm": int(retained_counts.min()),
        "maximum_selected_missingness": float(selected_missingness.max()),
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "leave_instance_stability": stability,
        "gates": gates,
        "feature_artifact_sha256": _sha256(args.features),
        "analysis_contract_sha256": _sha256(args.analysis_contract),
        "experiment_commit": str(frame["experiment_commit"].iloc[0]),
        "experiment_config_sha256": str(frame["experiment_config_sha256"].iloc[0]),
        "pyspi_config_sha256": str(frame["pyspi_config_sha256"].iloc[0]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "eligibility_pre_outcome.json").write_text(
        json.dumps(eligibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "fitted_target_blind_model.npz",
        feature_contract=np.asarray("unified_ordered_v3"),
        metric=np.asarray("pearson"),
        schema_sha256=np.asarray(payload["schema_sha256"]),
        spi_order=np.asarray(payload["spi_order"], dtype=str),
        keep_indices=transform.keep_indices,
        impute_values=transform.impute_values,
        center=transform.center,
        block_scale=transform.block_scale,
        components=components,
        explained_variance_ratio=pca.explained_variance_ratio_,
        q_center=np.asarray(q_center),
        q_scale=np.asarray(q_scale),
        minimum_valid_fraction=np.asarray(representation["minimum_valid_fraction"]),
        variance_threshold=np.asarray(representation["variance_threshold"]),
    )
    if not all(gates.values()):
        raise RuntimeError(f"Desai--Zwanzig analysis is ineligible: {gates}")

    # Outcome access begins only after the immutable target-blind eligibility record.
    targets = _load_targets(frame["path"].tolist())
    for column in targets:
        frame[column] = targets[column].to_numpy()
    frame = frame.loc[retained].reset_index(drop=True)
    fit = (
        frame["arm"].eq(representation["fit_arm"])
        & frame["instance"].isin(fit_instances)
    ).to_numpy()
    confirmation = frame["instance"].isin(confirmation_instances).to_numpy()
    mean_field_confirmation = confirmation & frame["arm"].eq("mean_field").to_numpy()
    finite_confirmation = confirmation & frame["arm"].eq("finite_N32").to_numpy()

    display_sign = 1.0
    if safe_spearman(frame.loc[fit, "q"], frame.loc[fit, "Q_mean_abs"]) < 0.0:
        display_sign = -1.0
    frame["q_display"] = display_sign * frame["q"]

    baselines = _input_baselines(frame["path"].tolist())
    for column in baselines:
        frame[column] = baselines[column].to_numpy()
    baseline_summary = {
        name: _association(frame, mean_field_confirmation, name)
        for name in baselines.columns
    }

    decoder = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(
        frame.loc[fit, "q_display"], frame.loc[fit, "Q_mean_abs"]
    )
    frame["Q_hat_q"] = decoder.predict(frame["q_display"])
    mae_draws = clustered_bootstrap_mae(
        frame.loc[mean_field_confirmation, "Q_mean_abs"],
        frame.loc[mean_field_confirmation, "Q_hat_q"],
        frame.loc[mean_field_confirmation, "instance"],
        n_resamples=2000,
        seed=29501,
    )

    primary_curve = frame.loc[mean_field_confirmation].groupby("sigma")[["q_display", "Q_mean_abs"]].mean()
    finite_curve = frame.loc[finite_confirmation].groupby("sigma")[["q_display", "Q_mean_abs"]].mean()
    q_boundary = _steepest_interval(primary_curve["q_display"])
    Q_boundary = _steepest_interval(primary_curve["Q_mean_abs"])
    q_sharpness = _normalized_maximum_adjacent_change(primary_curve["q_display"])
    Q_sharpness = _normalized_maximum_adjacent_change(primary_curve["Q_mean_abs"])
    reference = float(contract["physics"]["reference_mean_field_sigma_c"])
    summary = {
        "status": "independent_seed_confirmation_after_target_blind_local_fit",
        "representation_fit_uses_targets_or_controls": False,
        "display_sign_uses_development_Q_only": display_sign,
        "rows": int(len(frame)),
        "spi_count": int(np.asarray(payload["spi_order"]).size),
        "meta_feature_count": int(values.shape[1]),
        "selected_meta_feature_count": int(transform.keep_indices.size),
        "represented_spi_count": int(
            np.unique(
                np.concatenate(
                    [
                        np.asarray(payload["feature_spi_a"], dtype=str)[transform.keep_indices],
                        np.asarray(payload["feature_spi_b"], dtype=str)[transform.keep_indices],
                    ]
                )
            ).size
        ),
        "pc_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "mean_field_confirmation_association": _association(
            frame, mean_field_confirmation, "q_display"
        ),
        "finite_N32_confirmation_association": _association(
            frame, finite_confirmation, "q_display"
        ),
        "mean_field_boundary_localization": {
            "reference_mean_field_sigma_c": reference,
            "q": q_boundary,
            "Q": Q_boundary,
            "q_midpoint_minus_reference": float(q_boundary["midpoint"] - reference),
            "Q_midpoint_minus_reference": float(Q_boundary["midpoint"] - reference),
            "q_midpoint_minus_Q_midpoint": float(
                q_boundary["midpoint"] - Q_boundary["midpoint"]
            ),
        },
        "mean_field_normalized_sharpness": {
            "q": q_sharpness,
            "Q": Q_sharpness,
            "q_minus_Q_fraction_of_range": float(
                q_sharpness["fraction_of_total_range"]
                - Q_sharpness["fraction_of_total_range"]
            ),
        },
        "finite_N32_boundary_localization": {
            "q": _steepest_interval(finite_curve["q_display"]),
            "Q": _steepest_interval(finite_curve["Q_mean_abs"]),
        },
        "supervised_q_readout": {
            "mae": float(
                np.mean(
                    np.abs(
                        frame.loc[mean_field_confirmation, "Q_hat_q"]
                        - frame.loc[mean_field_confirmation, "Q_mean_abs"]
                    )
                )
            ),
            "mae_ci95": np.quantile(mae_draws, [0.025, 0.975]).tolist(),
        },
        "mean_field_input_baselines": baseline_summary,
        "future_Q_block_range_p95": float(frame["Q_block_range"].quantile(0.95)),
        "eligibility_precedes_outcome_access": True,
    }
    frame.drop(columns="path").to_csv(args.output_dir / "scores.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
