#!/usr/bin/env python3
"""Apply the frozen Stuart--Landau SPI--SPI coordinate to confirmation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_stuart_landau_development import (  # noqa: E402
    _association,
    _cell_summary,
    _input_baselines,
    _load_artifact,
    _paired_prefix_stability,
    _resolve_dataset_path,
)
from src.order_parameter_analysis import (  # noqa: E402
    clustered_bootstrap_difference,
    clustered_bootstrap_mae,
    clustered_bootstrap_spearman,
    safe_spearman,
)
from src.utils import load_json  # noqa: E402


EXPECTED_GAMMAS = {0.55, 0.65, 0.725, 0.775, 0.85, 0.95, 1.05, 1.15, 1.25}


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
        rows.append(
            {
                "path": path,
                "class_name": class_name,
                "arm": "full" if "full-observation" in class_name else "partial",
                "M": int(meta["M"]),
                "T": int(meta["T"]),
                "instance": int(meta["instance_index"]),
                "gamma": float(params["frequency_half_width"]),
                "coupling": float(params["coupling"]),
                "N_full": int(
                    meta["M"] if params.get("N_full") is None else params["N_full"]
                ),
                "experiment_commit": str(meta["experiment"]["git_commit"]),
                "experiment_config_sha256": str(meta["experiment"]["config_sha256"]),
                "pyspi_config_sha256": str(meta["pyspi"]["config_sha256"]),
                "pyspi_version": json.dumps(meta["pyspi"]["version"], sort_keys=True),
                "experiment_dirty": bool(meta["experiment"]["git_dirty"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["gamma_group"] = frame["gamma"].map(lambda value: f"{value:.6g}")
    return frame


def _load_targets(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        with np.load(path / "ground_truth.npz", allow_pickle=False) as truth:
            rows.append(
                {
                    "Q_R_mean": float(truth["q_R_mean"]),
                    "Q_R_sd": float(truth["q_R_std"]),
                    "Q_activity": float(truth["q_activity_mean"]),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_association(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    target: str = "Q_R_mean",
    bootstraps: int,
    seed: int,
) -> dict[str, object]:
    selected = frame.loc[mask]
    overall, within = clustered_bootstrap_spearman(
        selected["q"],
        selected[target],
        selected["gamma_group"],
        selected["instance"],
        n_resamples=bootstraps,
        seed=seed,
    )
    return {
        **_association(
            selected["q"].to_numpy(),
            selected[target].to_numpy(),
            selected["gamma_group"].to_numpy(),
        ),
        "overall_ci95": np.quantile(overall, [0.025, 0.975]).tolist(),
        "within_gamma_ci95": np.quantile(within, [0.025, 0.975]).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--development-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-arm", choices=("both", "full"), default="both")
    parser.add_argument(
        "--expected-gammas",
        default=",".join(map(str, sorted(EXPECTED_GAMMAS))),
        help="comma-separated confirmation controls",
    )
    parser.add_argument("--expected-M-values", default="8,16,32")
    parser.add_argument("--expected-T-values", default="100,500,1000")
    parser.add_argument("--expected-instances", type=int, default=8)
    parser.add_argument("--status-label")
    parser.add_argument("--maximum-selected-missingness", type=float, default=0.05)
    parser.add_argument("--exclude-rows-above-missingness", action="store_true")
    parser.add_argument("--maximum-excluded-fraction", type=float, default=0.02)
    parser.add_argument("--minimum-retained-per-control", type=int, default=6)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()

    payload = _load_artifact(args.features)
    values = np.asarray(payload["X"], dtype=np.float32)
    frame = _metadata_frame(payload, args.data_root)
    with np.load(args.model, allow_pickle=False) as archive:
        model = {name: archive[name] for name in archive.files}
    schema_matches = str(np.asarray(model["schema_sha256"]).item()) == str(
        np.asarray(payload["schema_sha256"]).item()
    )
    spi_order_matches = np.array_equal(
        np.asarray(model["spi_order"], dtype=str),
        np.asarray(payload["spi_order"], dtype=str),
    )
    keep = np.asarray(model["keep_indices"], dtype=np.int64)
    selected = values[:, keep]
    missingness = np.mean(~np.isfinite(selected), axis=1)
    filled = np.where(
        np.isfinite(selected), selected, np.asarray(model["impute_values"])
    )
    transformed = (
        filled - np.asarray(model["center"])
    ) * np.asarray(model["block_scale"])
    q_raw = (
        transformed @ np.asarray(model["components"])[0]
    ) * float(model["pc1_display_sign"])
    frame["q"] = (q_raw - float(model["q_center"])) / float(model["q_scale"])
    frame["selected_missingness"] = missingness

    if args.analysis_arm == "full":
        frame = frame.loc[frame["arm"].eq("full")].reset_index(drop=True)
        missingness = frame["selected_missingness"].to_numpy()

    expected_arms = (
        {"full", "partial"} if args.analysis_arm == "both" else {"full"}
    )
    expected_gammas = {float(value) for value in args.expected_gammas.split(",")}
    expected_M = {int(value) for value in args.expected_M_values.split(",")}
    expected_T = {int(value) for value in args.expected_T_values.split(",")}
    expected_instances = set(range(args.expected_instances))
    expected_rows = (
        len(expected_arms)
        * len(expected_gammas)
        * len(expected_M)
        * len(expected_T)
        * len(expected_instances)
    )

    design_valid = (
        len(frame) == expected_rows
        and set(frame["arm"]) == expected_arms
        and set(frame["M"]) == expected_M
        and set(frame["T"]) == expected_T
        and set(frame["instance"]) == expected_instances
        and set(frame["gamma"]) == expected_gammas
        and set(frame["coupling"]) == {0.8}
        and np.all(
            frame.loc[frame["arm"].eq("full"), "N_full"]
            == frame.loc[frame["arm"].eq("full"), "M"]
        )
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
    retained_mask = missingness <= args.maximum_selected_missingness
    retained_counts = (
        frame.assign(_retained=retained_mask)
        .groupby(["arm", "M", "T", "gamma"])["_retained"]
        .sum()
    )
    excluded_fraction = float(np.mean(~retained_mask))
    if args.exclude_rows_above_missingness:
        missingness_gate = (
            excluded_fraction <= args.maximum_excluded_fraction
            and int(retained_counts.min()) >= args.minimum_retained_per_control
        )
    else:
        missingness_gate = bool(retained_mask.all())
    gates = {
        "schema_matches_frozen_model": bool(schema_matches),
        "spi_order_matches_frozen_model": bool(spi_order_matches),
        "complete_confirmation_design": bool(design_valid),
        "homogeneous_clean_provenance": bool(provenance_valid),
        "all_retained_coordinates_finite": bool(
            np.isfinite(frame.loc[retained_mask, "q"]).all()
        ),
        "selected_missingness_policy_passes": bool(missingness_gate),
    }
    eligibility = {
        "status": "eligible" if all(gates.values()) else "ineligible",
        "outcomes_read": False,
        "analysis_arm": args.analysis_arm,
        "rows": int(len(frame)),
        "retained_rows": int(retained_mask.sum()),
        "excluded_rows": int((~retained_mask).sum()),
        "excluded_fraction": excluded_fraction,
        "excluded_row_keys": frame.loc[
            ~retained_mask, ["arm", "M", "T", "gamma", "instance"]
        ].to_dict(orient="records"),
        "minimum_retained_per_control": int(retained_counts.min()),
        "target_blind_row_exclusion_enabled": bool(
            args.exclude_rows_above_missingness
        ),
        "gates": gates,
        "maximum_selected_missingness": float(missingness.max()),
        "p99_selected_missingness": float(np.quantile(missingness, 0.99)),
        "feature_artifact_sha256": _sha256(args.features),
        "frozen_model_sha256": _sha256(args.model),
        "confirmation_controls": sorted(frame["gamma"].unique().tolist()),
        "experiment_commit": str(frame["experiment_commit"].iloc[0]),
        "experiment_config_sha256": str(frame["experiment_config_sha256"].iloc[0]),
        "pyspi_config_sha256": str(frame["pyspi_config_sha256"].iloc[0]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    eligibility_path = args.output_dir / "eligibility_pre_outcome.json"
    with eligibility_path.open("x", encoding="utf-8") as handle:
        json.dump(eligibility, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not all(gates.values()):
        raise RuntimeError(f"confirmation is ineligible: {gates}")

    if args.exclude_rows_above_missingness:
        frame = frame.loc[retained_mask].reset_index(drop=True)
        missingness = frame["selected_missingness"].to_numpy()

    # Outcome access starts here, after the immutable eligibility artifact.
    targets = _load_targets(frame["path"].tolist())
    for column in targets:
        frame[column] = targets[column].to_numpy()
    development = pd.read_csv(args.development_scores)
    supervised_fit = (
        development["arm"].eq("full")
        & development["T"].ge(500)
        & development["instance"].lt(4)
    )
    isotonic_q = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(
        development.loc[supervised_fit, "q"],
        development.loc[supervised_fit, "Q_R_mean"],
    )
    isotonic_control = IsotonicRegression(
        increasing="auto", out_of_bounds="clip"
    ).fit(
        development.loc[supervised_fit, "gamma"],
        development.loc[supervised_fit, "Q_R_mean"],
    )
    frame["Q_hat_q"] = isotonic_q.predict(frame["q"])
    frame["Q_hat_control"] = isotonic_control.predict(frame["gamma"])

    baseline = _input_baselines(frame["path"].tolist())
    for column in baseline:
        frame[column] = baseline[column].to_numpy()
    baseline_predictions = {}
    for column in baseline:
        model = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(
            development.loc[supervised_fit, column],
            development.loc[supervised_fit, "Q_R_mean"],
        )
        prediction_column = f"Q_hat_{column}"
        frame[prediction_column] = model.predict(frame[column])
        baseline_predictions[column] = prediction_column

    full = frame["arm"].eq("full").to_numpy()
    partial = frame["arm"].eq("partial").to_numpy()
    pooled = frame.loc[full].groupby("gamma")[["q", "Q_R_mean"]].mean()
    full_q_mae = float(np.mean(np.abs(frame.loc[full, "Q_hat_q"] - frame.loc[full, "Q_R_mean"])))
    full_control_mae = float(
        np.mean(
            np.abs(frame.loc[full, "Q_hat_control"] - frame.loc[full, "Q_R_mean"])
        )
    )
    mae_q_bootstrap = clustered_bootstrap_mae(
        frame.loc[full, "Q_R_mean"],
        frame.loc[full, "Q_hat_q"],
        frame.loc[full, "instance"],
        n_resamples=args.bootstraps,
        seed=8311,
    )
    q_vs_control_difference = clustered_bootstrap_difference(
        frame.loc[full, "Q_R_mean"],
        frame.loc[full, "Q_hat_q"],
        frame.loc[full, "Q_hat_control"],
        frame.loc[full, "instance"],
        n_resamples=args.bootstraps,
        seed=8312,
    )
    baseline_summary = {}
    for name, prediction_column in baseline_predictions.items():
        baseline_summary[name] = {
            **_association(
                frame.loc[full, name].to_numpy(),
                frame.loc[full, "Q_R_mean"].to_numpy(),
                frame.loc[full, "gamma_group"].to_numpy(),
            ),
            "supervised_isotonic_mae": float(
                np.mean(
                    np.abs(
                        frame.loc[full, prediction_column]
                        - frame.loc[full, "Q_R_mean"]
                    )
                )
            ),
        }
    summary = {
        "status": args.status_label or (
            "independent_confirmation"
            if args.analysis_arm == "both"
            else "independent_confirmation_full_arm_after_global_gate_failure"
        ),
        "eligibility_precedes_outcome_access": True,
        "rows": int(len(frame)),
        "confirmation_controls": sorted(frame["gamma"].unique().tolist()),
        "full_association": _bootstrap_association(
            frame, full, bootstraps=args.bootstraps, seed=8201
        ),
        "partial_association": (
            _bootstrap_association(
                frame, partial, bootstraps=args.bootstraps, seed=8202
            )
            if partial.any()
            else None
        ),
        "full_pooled_gamma_mean_spearman": safe_spearman(
            pooled["q"], pooled["Q_R_mean"]
        ),
        "full_R_sd_association": _bootstrap_association(
            frame,
            full,
            target="Q_R_sd",
            bootstraps=args.bootstraps,
            seed=8203,
        ),
        "full_pooled_gamma_mean_R_sd_spearman": safe_spearman(
            frame.loc[full].groupby("gamma")["q"].mean(),
            frame.loc[full].groupby("gamma")["Q_R_sd"].mean(),
        ),
        "full_cell_summary": _cell_summary(frame, full),
        "partial_cell_summary": _cell_summary(frame, partial) if partial.any() else None,
        "target_free_paired_prefix_stability": _paired_prefix_stability(
            frame, np.ones(len(frame), dtype=bool)
        ),
        "supervised_q_readout": {
            "mae": full_q_mae,
            "mae_ci95": np.quantile(mae_q_bootstrap, [0.025, 0.975]).tolist(),
        },
        "control_only_readout": {"mae": full_control_mae},
        "q_minus_control_mae_difference_ci95": np.quantile(
            q_vs_control_difference, [0.025, 0.975]
        ).tolist(),
        "input_baselines": baseline_summary,
        "maximum_selected_missingness": float(missingness.max()),
        "p99_selected_missingness": float(np.quantile(missingness, 0.99)),
        "frozen_model_sha256": eligibility["frozen_model_sha256"],
        "feature_artifact_sha256": eligibility["feature_artifact_sha256"],
    }
    frame.drop(columns="path").to_csv(args.output_dir / "scores.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
