#!/usr/bin/env python3
"""Retrospective full-p90 Kuramoto SPI--SPI order-coordinate reanalysis.

The representation is fitted without control or order-parameter values on the
disclosed benchmark bank plus the target-sealed eligibility-null bank. Only
Gaussian, randomly sampled natural-frequency datasets are used. Physical
targets are loaded after the frozen meta-feature transform and PC1 are fitted.
"""

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

ROOT = Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT / "src").is_dir():
    ROOT = ROOT.parent
if not (ROOT / "src").is_dir():
    ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.order_parameter_analysis import (  # noqa: E402
    clustered_bootstrap_mae,
    clustered_bootstrap_spearman,
    input_only_features,
    residualize_by_group,
    safe_spearman,
)
from src.spi_spi_analysis import fit_feature_transform  # noqa: E402
from src.utils import load_json  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_artifact(path: Path) -> dict[str, np.ndarray | str]:
    with np.load(path, allow_pickle=True) as archive:
        payload: dict[str, np.ndarray | str] = {
            name: archive[name] for name in archive.files
        }
    contract = str(np.asarray(payload["feature_contract"]).item())
    metric = str(np.asarray(payload["metric"]).item())
    subset = str(np.asarray(payload["spi_subset"]).item())
    spi_order = np.asarray(payload["spi_order"], dtype=str)
    values = np.asarray(payload["X"], dtype=np.float32)
    if contract != "unified_ordered_v3" or metric != "pearson":
        raise ValueError(f"{path}: expected unified_ordered_v3 Pearson artifact")
    if subset:
        raise ValueError(f"{path}: SPI subset was applied: {subset!r}")
    if spi_order.size != 289 or values.shape[1] != 289 * 288 // 2:
        raise ValueError(
            f"{path}: expected 289 SPIs/41,616 pairs, got "
            f"{spi_order.size}/{values.shape[1]}"
        )
    return payload


def _resolve_dataset_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _frame(payload: dict[str, np.ndarray | str], source: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_path in np.asarray(payload["dataset_paths"], dtype=object):
        dataset_path = _resolve_dataset_path(raw_path)
        meta = load_json(dataset_path / "meta.json")
        params = meta["generator"]["resolved_params"]
        sampling = meta["sampling_design"]
        distribution = str(params["frequency_distribution"])
        frequency_sampling = str(params["frequency_sampling"])
        if distribution != "gaussian" or frequency_sampling != "random":
            raise ValueError(
                f"{dataset_path}: expected Gaussian random frequencies, got "
                f"{distribution}/{frequency_sampling}"
            )
        role = str(sampling["role"])
        design = "paired" if role == "paired-control-path" else "cell"
        rows.append(
            {
                "path": dataset_path,
                "source": source,
                "class_name": str(meta["mts_class"]),
                "design": design,
                "kappa": float(meta["generator"]["control"]["reduced_value"]),
                "instance": int(meta["instance_index"]),
                "seed_group_id": str(sampling["seed_group_id"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["kappa_group"] = frame["kappa"].round(6).astype(str)
    frame["cluster"] = frame["class_name"] + ":" + frame["seed_group_id"]
    return frame


def _target(paths: list[Path]) -> np.ndarray:
    values = np.empty(len(paths), dtype=np.float64)
    for index, path in enumerate(paths):
        with np.load(path / "ground_truth.npz", allow_pickle=False) as truth:
            values[index] = float(np.mean(truth["r_full_future"]))
    return values


def _pc1(values: np.ndarray, random_state: int) -> tuple[np.ndarray, float]:
    fitted = PCA(
        n_components=1,
        svd_solver="randomized",
        iterated_power=7,
        random_state=int(random_state),
    ).fit(values)
    component = np.asarray(fitted.components_[0], dtype=np.float64)
    if component[np.argmax(np.abs(component))] < 0.0:
        component *= -1.0
    return component, float(fitted.explained_variance_ratio_[0])


def _source_stability(
    values: np.ndarray,
    source: np.ndarray,
    reference_component: np.ndarray,
    reference_coordinate: np.ndarray,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(np.unique(source)):
        component, variance = _pc1(values[source == name], 4400 + index)
        if np.dot(component, reference_component) < 0.0:
            component *= -1.0
        result[str(name)] = {
            "loading_cosine": float(np.dot(component, reference_component)),
            "coordinate_spearman": safe_spearman(
                values @ component, reference_coordinate
            ),
            "explained_variance_ratio": variance,
        }
    return result


def _association(
    coordinate: np.ndarray,
    target: np.ndarray,
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    n_bootstraps: int,
    seed: int,
) -> dict[str, object]:
    overall, within = clustered_bootstrap_spearman(
        coordinate[mask],
        target[mask],
        frame.loc[mask, "kappa_group"].to_numpy(),
        frame.loc[mask, "cluster"].to_numpy(),
        n_resamples=n_bootstraps,
        seed=seed,
    )
    residual_coordinate = residualize_by_group(
        coordinate[mask], frame.loc[mask, "kappa_group"].to_numpy()
    )
    residual_target = residualize_by_group(
        target[mask], frame.loc[mask, "kappa_group"].to_numpy()
    )
    return {
        "overall_spearman": safe_spearman(coordinate[mask], target[mask]),
        "overall_ci95": np.nanquantile(overall, [0.025, 0.975]).tolist(),
        "within_kappa_spearman": safe_spearman(
            residual_coordinate, residual_target
        ),
        "within_kappa_ci95": np.nanquantile(within, [0.025, 0.975]).tolist(),
    }


def _input_baselines(paths: list[Path]) -> dict[str, np.ndarray]:
    records = [input_only_features(np.load(path / "timeseries.npy")) for path in paths]
    return {
        "mean_abs_correlation": np.asarray(
            [record["mean_abs_correlation"] for record in records], dtype=np.float64
        ),
        "analytic_phase_coherence": np.asarray(
            [record["analytic_phase_coherence"] for record in records],
            dtype=np.float64,
        ),
        "temporal_spectral_entropy": np.asarray(
            [record["mean_temporal_spectral_entropy"] for record in records],
            dtype=np.float64,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-features", type=Path, required=True)
    parser.add_argument("--eligibility-null-features", type=Path, required=True)
    parser.add_argument("--terminal-features", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_full_catalogue_reanalysis",
    )
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.99)
    parser.add_argument("--variance-threshold", type=float, default=0.05)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()

    benchmark = _load_artifact(args.benchmark_features)
    eligibility_null = _load_artifact(args.eligibility_null_features)
    terminal = _load_artifact(args.terminal_features)
    schema_hashes = {
        str(np.asarray(payload["schema_sha256"]).item())
        for payload in (benchmark, eligibility_null, terminal)
    }
    spi_orders = {
        tuple(np.asarray(payload["spi_order"], dtype=str))
        for payload in (benchmark, eligibility_null, terminal)
    }
    if len(schema_hashes) != 1 or len(spi_orders) != 1:
        raise RuntimeError("feature artifacts do not share one full-p90 schema")

    benchmark_frame = _frame(benchmark, "disclosed_benchmark")
    null_frame = _frame(eligibility_null, "target_sealed_eligibility_null")
    terminal_frame = _frame(terminal, "terminal_reanalysis")
    development_frame = pd.concat(
        [benchmark_frame, null_frame], ignore_index=True
    )
    development = np.concatenate(
        [np.asarray(benchmark["X"]), np.asarray(eligibility_null["X"])], axis=0
    )
    terminal_values = np.asarray(terminal["X"])
    blocks = np.asarray(benchmark["feature_block"], dtype=str)

    transform = fit_feature_transform(
        development,
        blocks,
        minimum_valid_fraction=args.minimum_valid_fraction,
        variance_threshold=args.variance_threshold,
        block_balanced=False,
    )
    transformed_development = transform.transform(development)
    component, explained_variance = _pc1(transformed_development, 17239)
    development_coordinate = transformed_development @ component
    terminal_coordinate = transform.transform(terminal_values) @ component
    source_stability = _source_stability(
        transformed_development,
        development_frame["source"].to_numpy(),
        component,
        development_coordinate,
    )

    selected_terminal = terminal_values[:, transform.keep_indices]
    terminal_missingness = np.mean(~np.isfinite(selected_terminal), axis=1)
    selected_spi_a = np.asarray(benchmark["feature_spi_a"], dtype=str)[
        transform.keep_indices
    ]
    selected_spi_b = np.asarray(benchmark["feature_spi_b"], dtype=str)[
        transform.keep_indices
    ]
    represented_spis = np.unique(np.concatenate([selected_spi_a, selected_spi_b]))

    # Outcomes are deliberately loaded only after the representation is frozen.
    terminal_target = _target(terminal_frame["path"].tolist())
    primary = terminal_frame["design"].eq("paired").to_numpy()
    cell = terminal_frame["design"].eq("cell").to_numpy()
    primary_association = _association(
        terminal_coordinate,
        terminal_target,
        terminal_frame,
        primary,
        n_bootstraps=args.bootstraps,
        seed=5001,
    )
    cell_association = _association(
        terminal_coordinate,
        terminal_target,
        terminal_frame,
        cell,
        n_bootstraps=args.bootstraps,
        seed=5002,
    )

    old_target = _target(benchmark_frame["path"].tolist())
    old_primary = benchmark_frame["design"].eq("paired").to_numpy()
    terminal_primary_coordinate = terminal_coordinate[primary]
    isotonic = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(
        development_coordinate[: len(benchmark_frame)][old_primary],
        old_target[old_primary],
    )
    prediction = isotonic.predict(terminal_primary_coordinate)
    mae_bootstrap = clustered_bootstrap_mae(
        terminal_target[primary],
        prediction,
        terminal_frame.loc[primary, "cluster"].to_numpy(),
        n_resamples=args.bootstraps,
        seed=5003,
    )

    baseline_values = _input_baselines(terminal_frame["path"].tolist())
    baseline_associations = {
        name: {
            "overall_spearman": safe_spearman(values[primary], terminal_target[primary]),
            "within_kappa_spearman": safe_spearman(
                residualize_by_group(
                    values[primary], terminal_frame.loc[primary, "kappa_group"]
                ),
                residualize_by_group(
                    terminal_target[primary],
                    terminal_frame.loc[primary, "kappa_group"],
                ),
            ),
        }
        for name, values in baseline_values.items()
    }

    variance_curve = (
        pd.DataFrame(
            {
                "kappa": terminal_frame.loc[primary, "kappa"].to_numpy(),
                "q": terminal_coordinate[primary],
            }
        )
        .groupby("kappa")["q"]
        .var(ddof=1)
    )
    peak_kappa = float(variance_curve.idxmax())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "complete_retrospective_full_catalogue_reanalysis",
        "prospective": False,
        "outcomes_used_to_fit_representation": False,
        "outcomes_previously_disclosed": True,
        "frequency_distribution": "gaussian",
        "frequency_sampling": "random",
        "feature_contract": "unified_ordered_v3",
        "metric": "pearson",
        "catalogue_spis": 289,
        "catalogue_meta_features": 289 * 288 // 2,
        "development_rows": int(len(development_frame)),
        "terminal_rows": int(len(terminal_frame)),
        "minimum_valid_fraction": args.minimum_valid_fraction,
        "variance_threshold": args.variance_threshold,
        "retained_meta_features": int(transform.keep_indices.size),
        "represented_spis": int(represented_spis.size),
        "pc1_explained_variance_ratio": explained_variance,
        "source_stability": source_stability,
        "terminal_selected_feature_missingness_max": float(
            np.max(terminal_missingness)
        ),
        "terminal_selected_feature_missingness_p99": float(
            np.quantile(terminal_missingness, 0.99)
        ),
        "primary_association": primary_association,
        "independent_cell_association": cell_association,
        "supervised_isotonic_readout": {
            "mae": float(np.mean(np.abs(terminal_target[primary] - prediction))),
            "mae_ci95": np.nanquantile(mae_bootstrap, [0.025, 0.975]).tolist(),
        },
        "baseline_associations": baseline_associations,
        "target_free_q_variance_peak_kappa": peak_kappa,
        "schema_sha256": next(iter(schema_hashes)),
        "input_sha256": {
            "benchmark": _sha256(args.benchmark_features),
            "eligibility_null": _sha256(args.eligibility_null_features),
            "terminal": _sha256(args.terminal_features),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "model.npz",
        keep_indices=transform.keep_indices,
        impute_values=transform.impute_values,
        center=transform.center,
        component=component,
        explained_variance_ratio=np.asarray(explained_variance),
        spi_order=np.asarray(next(iter(spi_orders)), dtype=str),
        pair_spi_a=selected_spi_a,
        pair_spi_b=selected_spi_b,
        represented_spis=represented_spis,
        development_coordinate=development_coordinate,
        development_source=development_frame["source"].to_numpy(dtype=str),
    )
    np.savez_compressed(
        args.output_dir / "results.npz",
        class_name=terminal_frame["class_name"].to_numpy(dtype=str),
        design=terminal_frame["design"].to_numpy(dtype=str),
        kappa=terminal_frame["kappa"].to_numpy(dtype=np.float64),
        instance=terminal_frame["instance"].to_numpy(dtype=np.int32),
        coordinate_pc1=terminal_coordinate,
        target_full_future_R=terminal_target,
        selected_feature_missingness=terminal_missingness,
        prediction_primary=prediction,
        primary_row_indices=np.flatnonzero(primary),
        **baseline_values,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
