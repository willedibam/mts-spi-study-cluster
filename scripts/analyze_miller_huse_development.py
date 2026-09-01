#!/usr/bin/env python3
"""Audit full-p90 Miller--Huse recovery of future spin magnetization."""

from __future__ import annotations

import argparse
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
    _association,
    _input_baselines,
    _load_artifact,
    _resolve_dataset_path,
    _source_stability,
)
from src.order_parameter_analysis import safe_spearman  # noqa: E402
from src.spi_spi_analysis import fit_feature_transform  # noqa: E402
from src.utils import load_json  # noqa: E402


def _metadata_frame(payload: dict[str, np.ndarray], data_root: Path) -> pd.DataFrame:
    rows = []
    for raw in np.asarray(payload["dataset_paths"], dtype=object):
        path = _resolve_dataset_path(raw, data_root)
        meta = load_json(path / "meta.json")
        params = meta["generator"]["resolved_params"]
        rows.append(
            {
                "path": path,
                "arm": "distributed",
                "M": int(meta["M"]),
                "T": int(meta["T"]),
                "instance": int(meta["instance_index"]),
                "g": float(params["coupling"]),
                "mu": float(params["mu"]),
                "lattice_side": int(params["lattice_side"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["g_group"] = frame["g"].map(lambda value: f"{value:.6g}")
    return frame


def _targets(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        with np.load(path / "ground_truth.npz", allow_pickle=False) as truth:
            blocks = np.asarray(truth["q_spin_abs_blocks"], dtype=np.float64)
            rows.append(
                {
                    "Q_spin_abs": float(truth["q_spin_abs"]),
                    "Q_spin_rms": float(truth["q_spin_rms"]),
                    "Q_spin_abs_hidden": float(truth["q_spin_abs_unobserved"]),
                    "Q_block_mean_se": float(blocks.std(ddof=1) / np.sqrt(len(blocks))),
                    "Q_block_range": float(np.ptp(blocks)),
                    "Q_binder": float(truth["spin_binder_cumulant"]),
                    "Q_susceptibility": float(truth["spin_susceptibility"]),
                }
            )
    return pd.DataFrame(rows)


def _cell_summary(frame: pd.DataFrame, mask: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for M, group in frame.loc[mask].groupby("M"):
        means = group.groupby("g")[["q", "Q_spin_abs"]].mean()
        rows.append(
            {
                "M": int(M),
                "n": int(len(group)),
                **_association(
                    group["q"].to_numpy(),
                    group["Q_spin_abs"].to_numpy(),
                    group["g_group"].to_numpy(),
                ),
                "cell_mean_spearman": safe_spearman(
                    means["q"], means["Q_spin_abs"]
                ),
                "selected_missingness_mean": float(
                    group["selected_missingness"].mean()
                ),
                "selected_missingness_p95": float(
                    group["selected_missingness"].quantile(0.95)
                ),
                "Q_block_mean_se_median": float(group["Q_block_mean_se"].median()),
                "Q_block_mean_se_p95": float(group["Q_block_mean_se"].quantile(0.95)),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.99)
    parser.add_argument("--variance-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=62401)
    args = parser.parse_args()

    payload = _load_artifact(args.features)
    values = np.asarray(payload["X"], dtype=np.float32)
    frame = _metadata_frame(payload, args.data_root)
    fit_mask = frame["instance"].lt(4).to_numpy()
    held_mask = frame["instance"].ge(4).to_numpy()
    if (
        len(frame) != 288
        or int(fit_mask.sum()) != 144
        or set(frame["M"]) != {8, 16, 32}
        or set(frame["T"]) != {1000}
        or set(frame["mu"]) != {3.0}
        or set(frame["lattice_side"]) != {128}
    ):
        raise ValueError("unexpected Miller--Huse development design")

    # No control or physical target enters this representation fit.
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
    frame["selected_missingness"] = np.mean(
        ~np.isfinite(values[:, transform.keep_indices]), axis=1
    )

    # Physical outcomes are loaded only after target-blind q is fixed.
    target = _targets(frame["path"].tolist())
    for column in target:
        frame[column] = target[column].to_numpy()
    display_sign = 1.0
    if safe_spearman(scores[fit_mask, 0], frame.loc[fit_mask, "Q_spin_abs"]) < 0.0:
        display_sign = -1.0
    q_raw = display_sign * scores[:, 0]
    q_center = float(q_raw[fit_mask].mean())
    q_scale = float(q_raw[fit_mask].std())
    frame["q"] = (q_raw - q_center) / q_scale

    isotonic_q = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(
        frame.loc[fit_mask, "q"], frame.loc[fit_mask, "Q_spin_abs"]
    )
    isotonic_control = IsotonicRegression(
        increasing="auto", out_of_bounds="clip"
    ).fit(frame.loc[fit_mask, "g"], frame.loc[fit_mask, "Q_spin_abs"])
    frame["Q_hat_q"] = isotonic_q.predict(frame["q"])
    frame["Q_hat_control"] = isotonic_control.predict(frame["g"])

    baseline = _input_baselines(frame["path"].tolist())
    for column in baseline:
        frame[column] = baseline[column].to_numpy()
    held = frame.loc[held_mask]
    pooled = held.groupby("g")[["q", "Q_spin_abs"]].mean()
    oracle = []
    for index in range(component_count):
        oracle.append(
            {
                "component": index + 1,
                "explained_variance_ratio": float(pca.explained_variance_ratio_[index]),
                "held_abs_spearman": abs(
                    safe_spearman(scores[held_mask, index], held["Q_spin_abs"])
                ),
            }
        )
    baseline_summary = {
        column: _association(
            held[column].to_numpy(),
            held["Q_spin_abs"].to_numpy(),
            held["g_group"].to_numpy(),
        )
        for column in baseline
    }
    summary = {
        "status": "development_only",
        "representation_fit_uses_controls_or_physical_targets": False,
        "rows": int(len(frame)),
        "fit_rows": int(fit_mask.sum()),
        "held_rows": int(held_mask.sum()),
        "spi_count": int(np.asarray(payload["spi_order"]).size),
        "meta_feature_count": int(values.shape[1]),
        "selected_meta_feature_count": int(transform.keep_indices.size),
        "pc_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "held_association": _association(
            held["q"].to_numpy(),
            held["Q_spin_abs"].to_numpy(),
            held["g_group"].to_numpy(),
        ),
        "held_pooled_g_mean_spearman": safe_spearman(
            pooled["q"], pooled["Q_spin_abs"]
        ),
        "held_by_M": _cell_summary(frame, held_mask),
        "held_supervised_q_mae": float(
            np.mean(np.abs(held["Q_hat_q"] - held["Q_spin_abs"]))
        ),
        "held_control_only_mae": float(
            np.mean(np.abs(held["Q_hat_control"] - held["Q_spin_abs"]))
        ),
        "target_free_source_stability": _source_stability(
            transformed, frame, fit_mask, components[0], args.seed + 100
        ),
        "exploratory_oracle_pc": oracle,
        "held_input_baselines": baseline_summary,
        "truth_uncertainty": {
            "Q_block_mean_se_median": float(held["Q_block_mean_se"].median()),
            "Q_block_mean_se_p95": float(held["Q_block_mean_se"].quantile(0.95)),
            "Q_block_range_p95": float(held["Q_block_range"].quantile(0.95)),
            "full_hidden_difference_p95": float(
                np.quantile(
                    np.abs(held["Q_spin_abs"] - held["Q_spin_abs_hidden"]), 0.95
                )
            ),
        },
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
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
