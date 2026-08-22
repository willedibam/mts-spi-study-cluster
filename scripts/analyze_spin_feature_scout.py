#!/usr/bin/env python3
"""Target-blind stability audit for the kinetic-Ising SPI--SPI representation."""

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

from src.order_parameter_analysis import fit_frozen_pc1  # noqa: E402
from src.order_parameter_features import (  # noqa: E402
    build_meta_feature_matrix,
    stable_spi_names,
    validate_spi_catalogs,
)
from src.utils import load_json  # noqa: E402


EXPECTED_DATASETS = 360
REFERENCE_VIEW = (20, 1000)
GATES = {
    "minimum_core_spis": 100,
    "T500": {"coordinate": 0.85, "geometry": 0.80, "row": 0.85},
    "T2000": {"coordinate": 0.90, "geometry": 0.85, "row": 0.90},
    "M10": {"coordinate": 0.85, "geometry": 0.80, "row": 0.85},
    "M32": {"coordinate": 0.85, "geometry": 0.80, "row": 0.85},
    "bootstrap_pc1_loading_cosine": 0.90,
    "bootstrap_pc1_coordinate_spearman": 0.90,
    "loo_pc1_loading_cosine": 0.90,
    "loo_pc1_coordinate_spearman": 0.90,
}
N_BOOTSTRAPS = 500


def _rho(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3 or np.unique(first[valid]).size < 2 or np.unique(second[valid]).size < 2:
        return float("nan")
    return float(spearmanr(first[valid], second[valid]).statistic)


def _records(data_dir: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(data_dir.glob("*/*/meta.json")):
        meta = load_json(meta_path)
        params = meta["generator"]["resolved_params"]
        J_y = float(params["J_y"])
        rows.append(
            {
                "path": meta_path.parent,
                "physical_path": "isotropic" if np.isclose(J_y, 1.0) else "anisotropic",
                "control": float(params["reduced_coupling"]),
                "M": int(meta["M"]),
                "T": int(meta["T"]),
                "instance": int(meta["instance_index"]),
                "seed": int(meta["generator"]["seed"]),
                "seed_group_id": str(meta["sampling_design"]["seed_group_id"]),
                "config_sha256": str(meta["pyspi"]["config_sha256"]),
                "computation_version": str(meta["pyspi"]["version"]["computation"]),
                "compute_seconds": float(meta["job"]["compute_seconds"]),
                "n_errors": len(meta["pyspi"].get("errors", {})),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != EXPECTED_DATASETS:
        raise RuntimeError(f"incomplete scout: found {len(frame)}/{EXPECTED_DATASETS} datasets")
    if frame["config_sha256"].nunique() != 1 or frame["computation_version"].nunique() != 1:
        raise RuntimeError("pyspi configuration/version mismatch across datasets")
    for instance, group in frame.groupby("instance"):
        if group["seed"].nunique() != 1 or group["seed_group_id"].nunique() != 1:
            raise RuntimeError(f"master {instance} is not seed-paired across views")
    return frame.reset_index(drop=True)


def _aligned_view(
    frame: pd.DataFrame, M: int, T: int
) -> tuple[pd.DataFrame, np.ndarray]:
    keys = ["physical_path", "control", "instance"]
    group = frame[(frame["M"] == M) & (frame["T"] == T)].sort_values(keys)
    return group, group.index.to_numpy()


def _assert_nested_observations(frame: pd.DataFrame) -> None:
    reference, _ = _aligned_view(frame, *REFERENCE_VIEW)
    for _, row in reference.iterrows():
        selector = (
            (frame["physical_path"] == row["physical_path"])
            & np.isclose(frame["control"], row["control"])
            & (frame["instance"] == row["instance"])
        )
        views = {
            (int(candidate["M"]), int(candidate["T"])): candidate["path"]
            for _, candidate in frame[selector].iterrows()
        }
        arrays = {view: np.load(path / "timeseries.npy") for view, path in views.items()}
        if not np.array_equal(arrays[(20, 500)], arrays[(20, 1000)][:500]):
            raise RuntimeError("T=500 is not an exact prefix of T=1000")
        if not np.array_equal(arrays[(20, 1000)], arrays[(20, 2000)][:1000]):
            raise RuntimeError("T=1000 is not an exact prefix of T=2000")
        reference_path = views[(20, 1000)]
        with np.load(reference_path / "ground_truth.npz") as archive:
            reference_patch = np.asarray(archive["patch_indices"])
        reference_columns = {tuple(index): column for column, index in enumerate(reference_patch)}
        for M in (10, 32):
            with np.load(views[(M, 1000)] / "ground_truth.npz") as archive:
                patch = np.asarray(archive["patch_indices"])
            shared = [tuple(index) for index in patch if tuple(index) in reference_columns]
            reference_indices = [reference_columns[index] for index in shared]
            candidate_columns = {tuple(index): column for column, index in enumerate(patch)}
            candidate_indices = [candidate_columns[index] for index in shared]
            if len(shared) != min(M, 20) or not np.array_equal(
                arrays[(20, 1000)][:, reference_indices],
                arrays[(M, 1000)][:, candidate_indices],
            ):
                raise RuntimeError(f"M={M} is not the expected nested patch of M=20")


def _view_metrics(frame: pd.DataFrame, matrix: np.ndarray, coordinate: np.ndarray) -> pd.DataFrame:
    keys = ["physical_path", "control", "instance"]
    reference, reference_indices = _aligned_view(frame, *REFERENCE_VIEW)
    reference_indices = reference.index.to_numpy()
    reference_keys = list(map(tuple, reference[keys].to_numpy()))
    reference_distances = pdist(matrix[reference_indices])
    rows = []
    for (M, T), group in frame.groupby(["M", "T"]):
        group = group.sort_values(keys)
        if list(map(tuple, group[keys].to_numpy())) != reference_keys:
            raise RuntimeError(f"view M={M},T={T} does not share the reference masters")
        indices = group.index.to_numpy()
        row_correlations = [
            np.corrcoef(matrix[index], matrix[reference_index])[0, 1]
            for index, reference_index in zip(indices, reference_indices, strict=True)
        ]
        path_metrics = []
        for physical_path in np.unique(group["physical_path"]):
            path_mask = group["physical_path"].to_numpy() == physical_path
            path_metrics.append(
                (
                    _rho(coordinate[indices][path_mask], coordinate[reference_indices][path_mask]),
                    _rho(
                        pdist(matrix[indices][path_mask]),
                        pdist(matrix[reference_indices][path_mask]),
                    ),
                    float(np.nanmedian(np.asarray(row_correlations)[path_mask])),
                )
            )
        within_cell = []
        for physical_path in np.unique(group["physical_path"]):
            for control in np.unique(group["control"]):
                cell_mask = (
                    (group["physical_path"].to_numpy() == physical_path)
                    & np.isclose(group["control"].to_numpy(), control)
                )
                within_cell.append(
                    _rho(
                        coordinate[indices][cell_mask],
                        coordinate[reference_indices][cell_mask],
                    )
                )
        rows.append(
            {
                "M": int(M),
                "T": int(T),
                "coordinate_spearman": _rho(
                    coordinate[indices], coordinate[reference_indices]
                ),
                "geometry_spearman": _rho(
                    pdist(matrix[indices]), reference_distances
                ),
                "median_row_correlation": float(np.nanmedian(row_correlations)),
                "minimum_path_coordinate_spearman": float(
                    np.nanmin([metric[0] for metric in path_metrics])
                ),
                "minimum_path_geometry_spearman": float(
                    np.nanmin([metric[1] for metric in path_metrics])
                ),
                "minimum_path_median_row_correlation": float(
                    np.nanmin([metric[2] for metric in path_metrics])
                ),
                "minimum_within_cell_coordinate_spearman": float(np.nanmin(within_cell)),
                "median_within_cell_coordinate_spearman": float(np.nanmedian(within_cell)),
            }
        )
    return pd.DataFrame(rows).sort_values(["M", "T"]).reset_index(drop=True)


def _passes(row: pd.Series, gate: dict[str, float]) -> bool:
    return bool(
        row["bootstrap_p05_minimum_path_coordinate_spearman"] >= gate["coordinate"]
        and row["bootstrap_p05_minimum_path_geometry_spearman"] >= gate["geometry"]
        and row["bootstrap_p05_minimum_path_median_row_correlation"] >= gate["row"]
    )


def _view_bootstrap_lower_bounds(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    coordinate: np.ndarray,
    views: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(761239)
    reference, reference_indices = _aligned_view(frame, *REFERENCE_VIEW)
    outputs = []
    for _, view in views.iterrows():
        group, indices = _aligned_view(frame, int(view["M"]), int(view["T"]))
        row_correlations = np.asarray(
            [
                np.corrcoef(matrix[index], matrix[reference_index])[0, 1]
                for index, reference_index in zip(indices, reference_indices, strict=True)
            ]
        )
        instances = group["instance"].to_numpy()
        paths = group["physical_path"].to_numpy()
        masters = np.unique(instances)
        draws = np.empty((N_BOOTSTRAPS, 3), dtype=np.float64)
        for draw in range(N_BOOTSTRAPS):
            sampled = rng.choice(masters, size=len(masters), replace=True)
            positions = np.concatenate(
                [np.flatnonzero(instances == master) for master in sampled]
            )
            path_values = []
            for physical_path in np.unique(paths):
                selected = positions[paths[positions] == physical_path]
                path_values.append(
                    (
                        _rho(
                            coordinate[indices][selected],
                            coordinate[reference_indices][selected],
                        ),
                        _rho(
                            pdist(matrix[indices][selected]),
                            pdist(matrix[reference_indices][selected]),
                        ),
                        float(np.nanmedian(row_correlations[selected])),
                    )
                )
            draws[draw] = np.nanmin(np.asarray(path_values), axis=0)
        lower = np.nanquantile(draws, 0.05, axis=0)
        outputs.append(
            {
                "M": int(view["M"]),
                "T": int(view["T"]),
                "bootstrap_p05_minimum_path_coordinate_spearman": float(lower[0]),
                "bootstrap_p05_minimum_path_geometry_spearman": float(lower[1]),
                "bootstrap_p05_minimum_path_median_row_correlation": float(lower[2]),
            }
        )
    return views.merge(pd.DataFrame(outputs), on=["M", "T"], validate="one_to_one")


def _pc1_master_stability(
    selected_reference: np.ndarray,
    instances: np.ndarray,
    reference_component: np.ndarray,
    reference_coordinate: np.ndarray,
) -> dict[str, float]:
    masters = np.unique(instances)
    rng = np.random.default_rng(193771)

    def compare(indices: np.ndarray) -> tuple[float, float]:
        pca = PCA(n_components=1, svd_solver="full").fit(selected_reference[indices])
        component = pca.components_[0]
        cosine = float(np.dot(component, reference_component))
        if cosine < 0.0:
            component = -component
            cosine = -cosine
        coordinate = (selected_reference - pca.mean_) @ component
        return cosine, _rho(coordinate, reference_coordinate)

    bootstraps = np.empty((N_BOOTSTRAPS, 2), dtype=np.float64)
    for draw in range(N_BOOTSTRAPS):
        sampled = rng.choice(masters, size=len(masters), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(instances == master) for master in sampled]
        )
        bootstraps[draw] = compare(indices)
    leave_one_out = np.asarray(
        [compare(np.flatnonzero(instances != master)) for master in masters]
    )
    return {
        "bootstrap_p05_loading_cosine": float(np.quantile(bootstraps[:, 0], 0.05)),
        "bootstrap_p05_coordinate_spearman": float(
            np.quantile(bootstraps[:, 1], 0.05)
        ),
        "loo_minimum_loading_cosine": float(np.min(leave_one_out[:, 0])),
        "loo_minimum_coordinate_spearman": float(np.min(leave_one_out[:, 1])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kinetic_ising_feature_scout",
    )
    args = parser.parse_args()

    frame = _records(args.data_dir)
    _assert_nested_observations(frame)
    catalogs = validate_spi_catalogs(frame["path"].tolist())
    core_mask = frame["M"] >= 10
    core_dirs = frame.loc[core_mask, "path"].tolist()
    core_spis, rates = stable_spi_names(
        core_dirs, catalogs, min_valid_fraction=1.0
    )
    matrix, pairs = build_meta_feature_matrix(
        frame["path"].tolist(), catalogs, core_spis, metric="pearson"
    )
    reference_mask = (frame["M"] == REFERENCE_VIEW[0]) & (
        frame["T"] == REFERENCE_VIEW[1]
    )
    model = fit_frozen_pc1(matrix[reference_mask], variance_threshold=0.05)
    coordinate, row_missingness = model.transform(matrix)
    selected = matrix[:, model.feature_indices].copy()
    missing = ~np.isfinite(selected)
    if np.any(missing):
        rows, columns = np.where(missing)
        selected[rows, columns] = model.impute_values[columns]
    if np.any(row_missingness[core_mask] != 0.0):
        raise RuntimeError("the zero-failure M>=10 stress core produced missing meta-features")
    views = _view_metrics(frame, selected, coordinate)
    views = _view_bootstrap_lower_bounds(frame, selected, coordinate, views)
    reference_instances = frame.loc[reference_mask, "instance"].to_numpy()
    pc1_stability = _pc1_master_stability(
        selected[reference_mask],
        reference_instances,
        model.component,
        coordinate[reference_mask],
    )
    by_view = views.set_index(["M", "T"])
    gate_results = {
        "minimum_core_spis": len(core_spis) >= GATES["minimum_core_spis"],
        "T500": _passes(by_view.loc[(20, 500)], GATES["T500"]),
        "T2000": _passes(by_view.loc[(20, 2000)], GATES["T2000"]),
        "M10": _passes(by_view.loc[(10, 1000)], GATES["M10"]),
        "M32": _passes(by_view.loc[(32, 1000)], GATES["M32"]),
        "bootstrap_pc1_loading_cosine": (
            pc1_stability["bootstrap_p05_loading_cosine"]
            >= GATES["bootstrap_pc1_loading_cosine"]
        ),
        "bootstrap_pc1_coordinate_spearman": (
            pc1_stability["bootstrap_p05_coordinate_spearman"]
            >= GATES["bootstrap_pc1_coordinate_spearman"]
        ),
        "loo_pc1_loading_cosine": (
            pc1_stability["loo_minimum_loading_cosine"]
            >= GATES["loo_pc1_loading_cosine"]
        ),
        "loo_pc1_coordinate_spearman": (
            pc1_stability["loo_minimum_coordinate_spearman"]
            >= GATES["loo_pc1_coordinate_spearman"]
        ),
    }
    passed = all(gate_results.values())

    args.data_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "stable_spis.txt").write_text(
        "\n".join(core_spis) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.data_dir / "feature_scout.npz",
        **{column: views[column].to_numpy() for column in views.columns},
        pca_feature_indices=model.feature_indices,
        pca_center=model.center,
        pca_component=model.component,
    )
    summary = {
        "datasets": len(frame),
        "total_spis": len(catalogs),
        "core_selection_rows": len(core_dirs),
        "core_spis": len(core_spis),
        "minimum_core_validity_rate": float(min(rates[name] for name in core_spis)),
        "meta_features": len(pairs),
        "pca_features_retained": int(model.feature_indices.size),
        "reference_pc1_variance": model.explained_variance_ratio,
        "pc1_master_stability": pc1_stability,
        "maximum_row_missingness": float(np.max(row_missingness)),
        "median_compute_seconds": float(frame["compute_seconds"].median()),
        "p95_compute_seconds": float(frame["compute_seconds"].quantile(0.95)),
        "max_errors_per_dataset": int(frame["n_errors"].max()),
        "gates": GATES,
        "gate_results": gate_results,
        "passed": passed,
        "views": views.to_dict(orient="records"),
        "confirmation_missingness_rule": {
            "p99_row_missingness_max": 0.01,
            "maximum_row_missingness": 0.10,
            "frozen_zero_failure_core_coordinate_rank_agreement_min": 0.95,
        },
        "note": "No order-parameter values or control outcomes were used to select SPIs, fit PC1, or evaluate these stability gates.",
    }
    (args.data_dir / "feature_scout_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
