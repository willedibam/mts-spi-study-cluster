#!/usr/bin/env python3
"""Audit a full-p90 Stuart--Landau development feature artifact.

The SPI--SPI transform and PCA are fitted without controls or physical targets.
Targets are loaded only after the representation is fixed. Instances 0--3 form
the internal development split; instances 4--7 are an untouched seed check.
This is development analysis, not the final independent confirmation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.order_parameter_analysis import (  # noqa: E402
    input_only_features,
    residualize_by_group,
    safe_spearman,
)
from src.spi_spi_analysis import fit_feature_transform  # noqa: E402
from src.utils import load_json  # noqa: E402


def _scalar(payload: dict[str, np.ndarray], name: str) -> float:
    return float(np.asarray(payload[name]).item())


def _resolve_dataset_path(raw: object, data_root: Path) -> Path:
    path = Path(str(raw))
    if path.exists():
        return path
    candidate = data_root / path.parent.name / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"cannot resolve dataset path {path} under {data_root}")


def _load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}
    contract = str(np.asarray(payload["feature_contract"]).item())
    metric = str(np.asarray(payload["metric"]).item())
    subset = str(np.asarray(payload["spi_subset"]).item())
    spi_order = np.asarray(payload["spi_order"], dtype=str)
    values = np.asarray(payload["X"], dtype=np.float32)
    if contract != "unified_ordered_v3" or metric != "pearson":
        raise ValueError("expected unified_ordered_v3 Pearson features")
    if subset:
        raise ValueError(f"an SPI subset was applied: {subset!r}")
    if spi_order.size != 289 or values.shape[1] != 289 * 288 // 2:
        raise ValueError(
            f"expected 289 SPIs and 41,616 pairs, got {spi_order.size} and "
            f"{values.shape[1]}"
        )
    return payload


def _frame(payload: dict[str, np.ndarray], data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw in np.asarray(payload["dataset_paths"], dtype=object):
        path = _resolve_dataset_path(raw, data_root)
        meta = load_json(path / "meta.json")
        params = meta["generator"]["resolved_params"]
        truth_path = path / "ground_truth.npz"
        with np.load(truth_path, allow_pickle=False) as truth:
            q_mean = _scalar(truth, "q_R_mean")
            q_sd = _scalar(truth, "q_R_std")
            q_activity = _scalar(truth, "q_activity_mean")
        rows.append(
            {
                "path": path,
                "arm": (
                    "full"
                    if str(meta["mts_class"]).endswith("full-observation")
                    else "partial"
                ),
                "M": int(meta["M"]),
                "T": int(meta["T"]),
                "instance": int(meta["instance_index"]),
                "gamma": float(params["frequency_half_width"]),
                "coupling": float(params["coupling"]),
                "N_full": int(
                    meta["M"] if params.get("N_full") is None else params["N_full"]
                ),
                "seed_group": str(meta["sampling_design"]["seed_group_id"]),
                "Q_R_mean": q_mean,
                "Q_R_sd": q_sd,
                "Q_activity": q_activity,
            }
        )
    frame = pd.DataFrame(rows)
    if set(frame["arm"]) != {"full", "partial"}:
        raise ValueError("expected full- and partial-observation arms")
    if set(frame["coupling"]) != {0.8}:
        raise ValueError("expected the published fixed-K=0.8 path")
    frame["gamma_group"] = frame["gamma"].map(lambda value: f"{value:.6g}")
    return frame


def _anchor_components(components: np.ndarray) -> np.ndarray:
    anchored = np.asarray(components, dtype=np.float64).copy()
    for row in anchored:
        if row[int(np.argmax(np.abs(row)))] < 0.0:
            row *= -1.0
    return anchored


def _association(q: np.ndarray, target: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    return {
        "overall_spearman": safe_spearman(q, target),
        "within_gamma_spearman": safe_spearman(
            residualize_by_group(q, groups),
            residualize_by_group(target, groups),
        ),
    }


def _cell_summary(frame: pd.DataFrame, mask: np.ndarray) -> list[dict[str, object]]:
    columns = ["arm", "M", "T"]
    rows: list[dict[str, object]] = []
    for key, group in frame.loc[mask].groupby(columns, sort=True):
        association = _association(
            group["q"].to_numpy(),
            group["Q_R_mean"].to_numpy(),
            group["gamma_group"].to_numpy(),
        )
        cell_means = group.groupby("gamma", sort=True)[["q", "Q_R_mean"]].mean()
        arm, M, T = key
        rows.append(
            {
                "arm": str(arm),
                "M": int(M),
                "T": int(T),
                "n": int(len(group)),
                **association,
                "cell_mean_spearman": safe_spearman(
                    cell_means["q"], cell_means["Q_R_mean"]
                ),
                "selected_feature_missingness_mean": float(
                    group["selected_missingness"].mean()
                ),
                "selected_feature_missingness_p95": float(
                    group["selected_missingness"].quantile(0.95)
                ),
            }
        )
    return rows


def _source_stability(
    transformed: np.ndarray,
    frame: pd.DataFrame,
    fit_mask: np.ndarray,
    reference: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fit_frame = frame.loc[fit_mask]
    for index, ((M, T), labels) in enumerate(fit_frame.groupby(["M", "T"]).groups.items()):
        positions = frame.index.get_indexer(labels)
        pca = PCA(
            n_components=1,
            svd_solver="randomized",
            iterated_power=7,
            random_state=seed + index,
        ).fit(transformed[positions])
        component = _anchor_components(pca.components_)[0]
        if np.dot(component, reference) < 0.0:
            component *= -1.0
        rows.append(
            {
                "M": int(M),
                "T": int(T),
                "n": int(len(positions)),
                "loading_cosine": float(np.dot(component, reference)),
                "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            }
        )
    return rows


def _paired_prefix_stability(
    frame: pd.DataFrame,
    held_mask: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (arm, M), group in frame.loc[held_mask].groupby(["arm", "M"]):
        pivot = group.pivot(index=["gamma", "instance"], columns="T", values="q")
        if 1000 not in pivot:
            continue
        for T in (100, 500):
            if T not in pivot:
                continue
            rows.append(
                {
                    "arm": str(arm),
                    "M": int(M),
                    "T": int(T),
                    "reference_T": 1000,
                    "paired_spearman": safe_spearman(pivot[T], pivot[1000]),
                    "paired_mae": float(np.mean(np.abs(pivot[T] - pivot[1000]))),
                }
            )
    return rows


def _input_baselines(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        values = input_only_features(np.load(path / "timeseries.npy"))
        rows.append(
            {
                "mean_abs_correlation": values["mean_abs_correlation"],
                "analytic_phase_coherence": values["analytic_phase_coherence"],
                "temporal_spectral_entropy": values[
                    "mean_temporal_spectral_entropy"
                ],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.99)
    parser.add_argument("--variance-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=49211)
    args = parser.parse_args()

    payload = _load_artifact(args.features)
    values = np.asarray(payload["X"], dtype=np.float32)
    frame = _frame(payload, args.data_root)
    if len(frame) != len(values):
        raise ValueError("feature rows and metadata rows disagree")

    fit_mask = (
        frame["arm"].eq("full")
        & frame["T"].ge(500)
        & frame["instance"].lt(4)
    ).to_numpy()
    check_mask = frame["instance"].ge(4).to_numpy()
    if int(fit_mask.sum()) != 240 or int(check_mask.sum()) != 720:
        raise ValueError(
            f"expected 240 target-blind fit rows and 720 held-seed rows, got "
            f"{fit_mask.sum()} and {check_mask.sum()}"
        )

    # No target or control values are passed into either fit.
    transform = fit_feature_transform(
        values[fit_mask],
        np.asarray(payload["feature_block"], dtype=str),
        minimum_valid_fraction=args.minimum_valid_fraction,
        variance_threshold=args.variance_threshold,
        block_balanced=False,
    )
    transformed_fit = transform.transform(values[fit_mask])
    component_count = min(10, transformed_fit.shape[0] - 1, transformed_fit.shape[1])
    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        iterated_power=7,
        random_state=args.seed,
    ).fit(transformed_fit)
    components = _anchor_components(pca.components_)
    transformed = transform.transform(values)
    scores = transformed @ components.T

    # Component signs are arbitrary. Orient only PC1 for readable plots after
    # the target-blind representation has been completely fitted.
    fit_target = frame.loc[fit_mask, "Q_R_mean"].to_numpy()
    display_sign = 1.0
    if safe_spearman(scores[fit_mask, 0], fit_target) < 0.0:
        display_sign = -1.0
    q_raw = display_sign * scores[:, 0]
    q_center = float(np.mean(q_raw[fit_mask]))
    q_scale = float(np.std(q_raw[fit_mask]))
    frame["q"] = (q_raw - q_center) / q_scale
    selected = values[:, transform.keep_indices]
    frame["selected_missingness"] = np.mean(~np.isfinite(selected), axis=1)

    # These target-based quantities are explicitly exploratory diagnostics.
    oracle_pc = []
    held_primary = check_mask & frame["arm"].eq("full").to_numpy()
    for index in range(component_count):
        oracle_pc.append(
            {
                "component": index + 1,
                "explained_variance_ratio": float(pca.explained_variance_ratio_[index]),
                "held_seed_abs_spearman": abs(
                    safe_spearman(
                        scores[held_primary, index],
                        frame.loc[held_primary, "Q_R_mean"].to_numpy(),
                    )
                ),
            }
        )

    isotonic = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(
        frame.loc[fit_mask, "q"], frame.loc[fit_mask, "Q_R_mean"]
    )
    frame["Q_hat"] = isotonic.predict(frame["q"])
    baseline = _input_baselines(frame["path"].tolist())
    for column in baseline:
        frame[column] = baseline[column].to_numpy()

    held = frame.loc[held_primary]
    held_full_gamma_means = held.groupby("gamma")[["q", "Q_R_mean"]].mean()
    baseline_associations = {
        name: _association(
            held[name].to_numpy(),
            held["Q_R_mean"].to_numpy(),
            held["gamma_group"].to_numpy(),
        )
        for name in baseline.columns
    }
    summary = {
        "status": "development_only",
        "representation_fit_uses_targets_or_controls": False,
        "fit_rule": "full observation, T>=500, instances 0--3",
        "held_seed_rule": "instances 4--7",
        "rows": int(len(frame)),
        "fit_rows": int(fit_mask.sum()),
        "held_seed_rows": int(check_mask.sum()),
        "spi_count": int(np.asarray(payload["spi_order"]).size),
        "meta_feature_count": int(values.shape[1]),
        "selected_meta_feature_count": int(transform.keep_indices.size),
        "represented_spi_count": int(
            np.unique(
                np.concatenate(
                    [
                        np.asarray(payload["feature_spi_a"], dtype=str)[
                            transform.keep_indices
                        ],
                        np.asarray(payload["feature_spi_b"], dtype=str)[
                            transform.keep_indices
                        ],
                    ]
                )
            ).size
        ),
        "minimum_valid_fraction": args.minimum_valid_fraction,
        "variance_threshold": args.variance_threshold,
        "pc_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc1_display_sign": display_sign,
        "held_full_association": _association(
            held["q"].to_numpy(),
            held["Q_R_mean"].to_numpy(),
            held["gamma_group"].to_numpy(),
        ),
        "held_full_pooled_gamma_mean_spearman": safe_spearman(
            held_full_gamma_means["q"], held_full_gamma_means["Q_R_mean"]
        ),
        "held_partial_association": _association(
            frame.loc[check_mask & frame["arm"].eq("partial"), "q"].to_numpy(),
            frame.loc[
                check_mask & frame["arm"].eq("partial"), "Q_R_mean"
            ].to_numpy(),
            frame.loc[
                check_mask & frame["arm"].eq("partial"), "gamma_group"
            ].to_numpy(),
        ),
        "held_full_supervised_isotonic_mae": float(
            np.mean(np.abs(held["Q_hat"] - held["Q_R_mean"]))
        ),
        "held_full_cell_summary": _cell_summary(frame, held_primary),
        "held_all_cell_summary": _cell_summary(frame, check_mask),
        "target_free_source_stability": _source_stability(
            transformed,
            frame,
            fit_mask,
            components[0],
            args.seed + 100,
        ),
        "held_target_free_paired_prefix_stability": _paired_prefix_stability(
            frame, check_mask
        ),
        "exploratory_oracle_pc": oracle_pc,
        "held_full_input_baselines": baseline_associations,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.drop(columns="path").to_csv(args.output_dir / "scores.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        pc1_display_sign=np.asarray(display_sign),
        minimum_valid_fraction=np.asarray(args.minimum_valid_fraction),
        variance_threshold=np.asarray(args.variance_threshold),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
