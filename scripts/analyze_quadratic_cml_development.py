#!/usr/bin/env python3
"""Audit quadratic-CML SPI--SPI recovery of a physical regime vector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

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


Q_COLUMNS = [
    "Q_temporal_entropy",
    "Q_pattern_entropy",
    "Q_selected_band_power",
    "Q_period2_residual",
]


def _resolve_path(raw: object, data_root: Path) -> Path:
    path = Path(str(raw))
    if path.exists():
        return path
    candidate = data_root / path.parent.name / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"cannot resolve {path} under {data_root}")


def _load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}
    values = np.asarray(payload["X"], dtype=np.float32)
    if str(np.asarray(payload["feature_contract"]).item()) != "unified_ordered_v3":
        raise ValueError("expected unified_ordered_v3 features")
    if str(np.asarray(payload["metric"]).item()) != "pearson":
        raise ValueError("expected Pearson SPI--SPI features")
    if str(np.asarray(payload["spi_subset"]).item()):
        raise ValueError("an SPI subset was applied")
    if np.asarray(payload["spi_order"]).size != 289 or values.shape[1] != 41616:
        raise ValueError("expected the complete 289-SPI / 41,616-pair catalogue")
    return payload


def _frame(payload: dict[str, np.ndarray], data_root: Path) -> pd.DataFrame:
    rows = []
    for raw in np.asarray(payload["dataset_paths"], dtype=object):
        path = _resolve_path(raw, data_root)
        meta = load_json(path / "meta.json")
        params = meta["generator"]["resolved_params"]
        with np.load(path / "ground_truth.npz", allow_pickle=False) as truth:
            values = {
                "Q_temporal_entropy": float(truth["q_temporal_spectral_entropy"]),
                "Q_pattern_entropy": float(
                    truth["q_dynamical_spatial_pattern_entropy"]
                ),
                "Q_selected_band_power": float(truth["q_selected_band_power"]),
                "Q_period2_residual": float(truth["q_period2_activity"]),
            }
        class_name = str(meta["mts_class"])
        rows.append(
            {
                "path": path,
                "class_name": class_name,
                "arm": "large" if class_name.endswith("large-lattice") else "small-full",
                "M": int(meta["M"]),
                "T": int(meta["T"]),
                "instance": int(meta["instance_index"]),
                "alpha": float(params["alpha"]),
                "eps": float(params["eps"]),
                "N_full": int(params["lattice_size"]),
                **values,
            }
        )
    frame = pd.DataFrame(rows)
    if set(frame["eps"]) != {0.3} or set(frame["T"]) != {1000}:
        raise ValueError("expected the fixed-eps=.3, T=1000 development path")
    frame["alpha_group"] = frame["alpha"].map(lambda value: f"{value:.6g}")
    return frame


def _anchor(components: np.ndarray) -> np.ndarray:
    result = np.asarray(components, dtype=np.float64).copy()
    for row in result:
        if row[int(np.argmax(np.abs(row)))] < 0.0:
            row *= -1.0
    return result


def _association(values: np.ndarray, truth: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    return {
        "overall_spearman": safe_spearman(values, truth),
        "within_alpha_spearman": safe_spearman(
            residualize_by_group(values, groups),
            residualize_by_group(truth, groups),
        ),
    }


def _distance_association(q: np.ndarray, Q: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    overall = safe_spearman(pdist(q), pdist(Q))
    q_within, Q_within = [], []
    for group in np.unique(groups):
        member = groups == group
        if int(member.sum()) >= 3:
            q_within.extend(pdist(q[member]))
            Q_within.extend(pdist(Q[member]))
    return {
        "overall_pairwise_distance_spearman": overall,
        "within_alpha_pairwise_distance_spearman": safe_spearman(q_within, Q_within),
    }


def _source_stability(
    transformed_fit: np.ndarray,
    fit_frame: pd.DataFrame,
    reference: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    for offset, (M, labels) in enumerate(fit_frame.groupby("M").groups.items()):
        positions = fit_frame.index.get_indexer(labels)
        model = PCA(
            n_components=1,
            svd_solver="randomized",
            iterated_power=7,
            random_state=seed + offset,
        ).fit(transformed_fit[positions])
        component = _anchor(model.components_)[0]
        if np.dot(component, reference) < 0.0:
            component *= -1.0
        rows.append(
            {
                "M": int(M),
                "loading_cosine": float(np.dot(component, reference)),
                "explained_variance_ratio": float(model.explained_variance_ratio_[0]),
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
    parser.add_argument("--seed", type=int, default=77109)
    args = parser.parse_args()

    payload = _load_artifact(args.features)
    values = np.asarray(payload["X"], dtype=np.float32)
    frame = _frame(payload, args.data_root)
    fit_mask = (
        frame["arm"].eq("large") & frame["instance"].lt(4)
    ).to_numpy()
    held_large = (
        frame["arm"].eq("large") & frame["instance"].ge(4)
    ).to_numpy()
    held_all = frame["instance"].ge(4).to_numpy()
    large_controls = frame.loc[frame["arm"].eq("large"), "alpha"].nunique()
    small_controls = frame.loc[frame["arm"].eq("small-full"), "alpha"].nunique()
    expected_fit = 3 * 4 * large_controls
    expected_rows = 3 * 8 * (large_controls + small_controls)
    if (
        large_controls != 41
        or small_controls != 10
        or len(frame) != expected_rows
        or int(fit_mask.sum()) != expected_fit
        or int(held_large.sum()) != expected_fit
    ):
        raise ValueError("unexpected quadratic-CML development design")

    # The SPI representation is fitted without alpha or the physical vector.
    transform = fit_feature_transform(
        values[fit_mask],
        np.asarray(payload["feature_block"], dtype=str),
        minimum_valid_fraction=args.minimum_valid_fraction,
        variance_threshold=args.variance_threshold,
        block_balanced=False,
    )
    transformed_fit = transform.transform(values[fit_mask])
    component_count = min(10, transformed_fit.shape[0] - 1, transformed_fit.shape[1])
    q_pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        iterated_power=7,
        random_state=args.seed,
    ).fit(transformed_fit)
    q_components = _anchor(q_pca.components_)
    transformed = transform.transform(values)
    q_scores = transformed @ q_components.T
    for index in range(min(3, component_count)):
        frame[f"q{index + 1}"] = q_scores[:, index]
    frame["selected_missingness"] = np.mean(
        ~np.isfinite(values[:, transform.keep_indices]), axis=1
    )

    # The physical-vector reduction is part of outcome evaluation, not the
    # unsupervised q fit. It is recorded separately to avoid a scalar-Q claim.
    Q_fit = frame.loc[fit_mask, Q_COLUMNS].to_numpy()
    Q_center = Q_fit.mean(axis=0)
    Q_scale = Q_fit.std(axis=0)
    Q_standard = (frame[Q_COLUMNS].to_numpy() - Q_center) / Q_scale
    Q_pca = PCA(n_components=2, svd_solver="full").fit(Q_standard[fit_mask])
    Q_scores = Q_pca.transform(Q_standard)
    frame["Q_phys1"] = Q_scores[:, 0]
    frame["Q_phys2"] = Q_scores[:, 1]

    # Explicitly supervised multivariate readout, fit after q is frozen.
    decoder = Ridge(alpha=1.0).fit(q_scores[fit_mask, :5], Q_standard[fit_mask])
    Q_hat = decoder.predict(q_scores[:, :5])
    frame["Q_hat_mae"] = np.mean(np.abs(Q_hat - Q_standard), axis=1)

    held_groups = frame.loc[held_large, "alpha_group"].to_numpy()
    q_component_associations = {}
    for pc in range(min(3, component_count)):
        q_component_associations[f"q{pc + 1}"] = {
            target: _association(
                q_scores[held_large, pc],
                frame.loc[held_large, target].to_numpy(),
                held_groups,
            )
            for target in Q_COLUMNS
        }
    by_M = []
    for M, group in frame.loc[held_large].groupby("M"):
        indices = group.index.to_numpy()
        by_M.append(
            {
                "M": int(M),
                **_distance_association(
                    q_scores[indices, :2],
                    Q_standard[indices],
                    group["alpha_group"].to_numpy(),
                ),
                "supervised_vector_mae": float(frame.loc[indices, "Q_hat_mae"].mean()),
                "selected_missingness_p95": float(
                    frame.loc[indices, "selected_missingness"].quantile(0.95)
                ),
            }
        )

    baselines = []
    for path in frame["path"]:
        record = input_only_features(np.load(path / "timeseries.npy"))
        baselines.append(
            {
                "mean_abs_correlation": record["mean_abs_correlation"],
                "temporal_spectral_entropy": record[
                    "mean_temporal_spectral_entropy"
                ],
            }
        )
    for column in baselines[0]:
        frame[column] = [row[column] for row in baselines]

    baseline_columns = ["mean_abs_correlation", "temporal_spectral_entropy"]
    baseline_fit = frame.loc[fit_mask, baseline_columns].to_numpy()
    baseline_center = baseline_fit.mean(axis=0)
    baseline_scale = baseline_fit.std(axis=0)
    baseline_scale[baseline_scale == 0] = 1.0
    baseline_standard = (
        frame[baseline_columns].to_numpy() - baseline_center
    ) / baseline_scale
    baseline_decoder = Ridge(alpha=1.0).fit(
        baseline_standard[fit_mask], Q_standard[fit_mask]
    )
    baseline_hat = baseline_decoder.predict(baseline_standard)
    frame["Q_hat_input_baseline_mae"] = np.mean(
        np.abs(baseline_hat - Q_standard), axis=1
    )

    # Same-control development means form an explicit labelled control-only
    # comparator. It is evaluation, never part of the q representation.
    fit_control = frame.loc[fit_mask, "alpha"].to_numpy()
    control_grid = np.unique(fit_control)
    control_truth = np.stack(
        [Q_standard[fit_mask][fit_control == alpha].mean(axis=0) for alpha in control_grid]
    )
    control_hat = np.column_stack(
        [
            np.interp(frame["alpha"].to_numpy(), control_grid, control_truth[:, index])
            for index in range(len(Q_COLUMNS))
        ]
    )
    frame["Q_hat_control_mae"] = np.mean(
        np.abs(control_hat - Q_standard), axis=1
    )

    baseline_associations = {
        column: _association(
            frame.loc[held_large, column].to_numpy(),
            frame.loc[held_large, "Q_phys1"].to_numpy(),
            held_groups,
        )
        for column in baseline_columns
    }

    summary = {
        "status": "development_only_noncanonical_vector_Q",
        "representation_fit_uses_controls_or_physical_targets": False,
        "fit_rule": "N=512 large-lattice arm, instances 0--3",
        "held_rule": "instances 4--7",
        "rows": int(len(frame)),
        "fit_rows": int(fit_mask.sum()),
        "held_large_rows": int(held_large.sum()),
        "held_all_rows": int(held_all.sum()),
        "spi_count": int(np.asarray(payload["spi_order"]).size),
        "meta_feature_count": int(values.shape[1]),
        "selected_meta_feature_count": int(transform.keep_indices.size),
        "q_explained_variance_ratio": q_pca.explained_variance_ratio_.tolist(),
        "physical_vector_columns": Q_COLUMNS,
        "physical_vector_pca_explained_variance_ratio": (
            Q_pca.explained_variance_ratio_.tolist()
        ),
        "held_large_q_component_associations": q_component_associations,
        "held_large_q1_vs_physical_pc1": _association(
            q_scores[held_large, 0],
            frame.loc[held_large, "Q_phys1"].to_numpy(),
            held_groups,
        ),
        "held_large_input_baseline_associations_with_physical_pc1": (
            baseline_associations
        ),
        "held_large_two_dimensional_geometry": _distance_association(
            q_scores[held_large, :2], Q_standard[held_large], held_groups
        ),
        "held_large_supervised_vector_mae": float(
            frame.loc[held_large, "Q_hat_mae"].mean()
        ),
        "held_large_input_baseline_vector_mae": float(
            frame.loc[held_large, "Q_hat_input_baseline_mae"].mean()
        ),
        "held_large_control_only_vector_mae": float(
            frame.loc[held_large, "Q_hat_control_mae"].mean()
        ),
        "held_large_by_M": by_M,
        "target_free_loading_stability_by_M": _source_stability(
            transformed_fit,
            frame.loc[fit_mask].reset_index(drop=True),
            q_components[0],
            args.seed + 100,
        ),
        "small_full_finite_size_geometry": _distance_association(
            q_scores[held_all & frame["arm"].eq("small-full").to_numpy(), :2],
            Q_standard[held_all & frame["arm"].eq("small-full").to_numpy()],
            frame.loc[
                held_all & frame["arm"].eq("small-full").to_numpy(), "alpha_group"
            ].to_numpy(),
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.drop(columns="path").to_csv(args.output_dir / "scores.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "fitted_target_blind_model.npz",
        keep_indices=transform.keep_indices,
        impute_values=transform.impute_values,
        center=transform.center,
        block_scale=transform.block_scale,
        q_components=q_components,
        q_explained_variance_ratio=q_pca.explained_variance_ratio_,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
