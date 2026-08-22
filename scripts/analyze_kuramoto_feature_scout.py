#!/usr/bin/env python3
"""Audit SPI stability and nested-view geometry for the Kuramoto feature scout."""

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

from src.order_parameter_features import (  # noqa: E402
    build_meta_feature_matrix,
    explicit_phase_spi_names,
    stable_spi_names,
    validate_spi_catalogs,
)
from src.utils import load_json  # noqa: E402


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _records(data_dir: Path) -> pd.DataFrame:
    rows = []
    for meta_path in sorted(data_dir.glob("*/*/meta.json")):
        meta = load_json(meta_path)
        params = meta["generator"]["resolved_params"]
        truth_path = meta_path.parent / meta["generator"]["ground_truth"]["path"]
        with np.load(truth_path) as truth:
            r_full = float(np.mean(truth["r_full"]))
            r_hidden = (
                float(np.mean(truth["r_unobserved"]))
                if "r_unobserved" in truth.files
                else float("nan")
            )
        rows.append(
            {
                "path": meta_path.parent,
                "M": int(meta["M"]),
                "T": int(meta["T"]),
                "instance": int(meta["instance_index"]),
                "kappa": float(params["K"]) / float(meta["generator"]["ground_truth"]["critical_coupling"]),
                "zscore": bool(params["zscore"]),
                "r_full": r_full,
                "r_hidden": r_hidden,
                "compute_seconds": float(meta["job"]["compute_seconds"]),
                "n_errors": len(meta["pyspi"].get("errors", {})),
            }
        )
    if not rows:
        raise RuntimeError(f"no completed datasets under {data_dir}")
    return pd.DataFrame(rows)


def _fit_coordinate(
    X_fit: np.ndarray,
    X: np.ndarray,
    *,
    variance_threshold: float,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    keep = np.isfinite(X_fit).all(axis=0) & (np.std(X_fit, axis=0) >= variance_threshold)
    if not np.any(keep):
        raise RuntimeError("no finite varying meta-features remain")
    pca = PCA(n_components=1, svd_solver="full").fit(X_fit[:, keep])
    return pca.transform(X[:, keep])[:, 0], keep, pca


def _view_summary(frame: pd.DataFrame, X: np.ndarray, q: np.ndarray) -> pd.DataFrame:
    frame = frame.copy()
    frame["q"] = q
    keys = ["kappa", "instance"]
    reference = frame[(frame["M"] == 32) & (frame["T"] == 2000)].sort_values(keys)
    if len(reference) != frame.groupby(["M", "T"]).size().max():
        raise RuntimeError("reference M32/T2000 view is incomplete")
    reference_indices = reference.index.to_numpy()
    q_reference = reference["q"].to_numpy()
    distance_reference = pdist(X[reference_indices])
    rows = []
    for (M, T), group in frame.groupby(["M", "T"]):
        group = group.sort_values(keys)
        if list(map(tuple, group[keys].to_numpy())) != list(
            map(tuple, reference[keys].to_numpy())
        ):
            raise RuntimeError(f"nested-view keys do not align for M={M}, T={T}")
        indices = group.index.to_numpy()
        q_values = group["q"].to_numpy()
        r_residual = group["r_hidden"] - group.groupby("kappa")["r_hidden"].transform("mean")
        q_residual = group["q"] - group.groupby("kappa")["q"].transform("mean")
        row_correlations = [
            np.corrcoef(X[i], X[j])[0, 1]
            for i, j in zip(indices, reference_indices)
        ]
        rows.append(
            {
                "M": int(M),
                "T": int(T),
                "q_vs_reference_spearman": _rho(q_values, q_reference),
                "geometry_vs_reference_spearman": _rho(
                    pdist(X[indices]), distance_reference
                ),
                "median_feature_vector_correlation": float(np.nanmedian(row_correlations)),
                "q_vs_r_full_spearman": _rho(q_values, group["r_full"].to_numpy()),
                "q_vs_r_hidden_spearman": _rho(q_values, group["r_hidden"].to_numpy()),
                "within_kappa_q_vs_r_hidden_spearman": _rho(
                    q_residual.to_numpy(), r_residual.to_numpy()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["M", "T"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_feature_scout",
    )
    parser.add_argument("--variance-threshold", type=float, default=0.05)
    args = parser.parse_args()

    frame = _records(args.data_dir).reset_index(drop=True)
    raw = frame[~frame["zscore"]].copy().reset_index(drop=True)
    zscored = frame[frame["zscore"]].copy().reset_index(drop=True)
    expected_raw = 3 * 3 * 3 * 4
    expected_zscored = 3 * 4
    if len(raw) != expected_raw or len(zscored) != expected_zscored:
        raise RuntimeError(
            f"incomplete scout: raw={len(raw)}/{expected_raw}, "
            f"zscored={len(zscored)}/{expected_zscored}"
        )

    all_dirs = frame["path"].tolist()
    catalog = validate_spi_catalogs(all_dirs)
    stable, rates = stable_spi_names(all_dirs, catalog, min_valid_fraction=1.0)
    phase_names = explicit_phase_spi_names(catalog)
    stable_no_phase = [name for name in stable if name not in phase_names]

    X, pairs = build_meta_feature_matrix(
        all_dirs, catalog, stable, metric="pearson"
    )
    raw_X = X[~frame["zscore"].to_numpy()]
    z_X = X[frame["zscore"].to_numpy()]
    fit_mask = (raw["M"] == 32) & (raw["T"] == 2000)
    q_raw, keep, pca = _fit_coordinate(
        raw_X[fit_mask], raw_X, variance_threshold=args.variance_threshold
    )
    orientation = np.sign(_rho(q_raw[fit_mask], raw.loc[fit_mask, "kappa"].to_numpy())) or 1.0
    q_raw *= orientation
    views = _view_summary(raw, raw_X[:, keep], q_raw)

    # The z-score transform is evaluated through the exact raw-fitted loading.
    q_z = pca.transform(z_X[:, keep])[:, 0] * orientation
    raw_primary = raw[(raw["M"] == 20) & (raw["T"] == 1000)].sort_values(
        ["kappa", "instance"]
    )
    z_primary = zscored.sort_values(["kappa", "instance"])
    raw_q_primary = q_raw[raw_primary.index]
    z_q_ordered = q_z[z_primary.index]

    X_no_phase, _ = build_meta_feature_matrix(
        raw["path"].tolist(), catalog, stable_no_phase, metric="pearson"
    )
    q_no_phase, keep_no_phase, pca_no_phase = _fit_coordinate(
        X_no_phase[fit_mask], X_no_phase, variance_threshold=args.variance_threshold
    )
    q_no_phase *= np.sign(
        _rho(q_no_phase[fit_mask], raw.loc[fit_mask, "kappa"].to_numpy())
    ) or 1.0

    views.to_csv(args.data_dir / "feature_scout_views.csv", index=False)
    (args.data_dir / "stable_spis.txt").write_text(
        "\n".join(stable) + "\n", encoding="utf-8"
    )
    summary = {
        "datasets": int(len(frame)),
        "stable_spis": len(stable),
        "total_spis": len(catalog),
        "explicit_phase_spis_removed": len(set(stable) & phase_names),
        "stable_spis_no_phase": len(stable_no_phase),
        "meta_features": len(pairs),
        "meta_features_retained": int(keep.sum()),
        "reference_pc1_variance": float(pca.explained_variance_ratio_[0]),
        "no_phase_meta_features_retained": int(keep_no_phase.sum()),
        "no_phase_reference_pc1_variance": float(pca_no_phase.explained_variance_ratio_[0]),
        "raw_vs_zscore_q_spearman": _rho(raw_q_primary, z_q_ordered),
        "zscore_q_vs_r_hidden_spearman": _rho(
            z_q_ordered, z_primary["r_hidden"].to_numpy()
        ),
        "raw_primary_q_vs_no_phase_q_spearman": _rho(
            raw_q_primary, q_no_phase[raw_primary.index]
        ),
        "median_compute_seconds": float(frame["compute_seconds"].median()),
        "p95_compute_seconds": float(frame["compute_seconds"].quantile(0.95)),
        "max_compute_seconds": float(frame["compute_seconds"].max()),
        "max_errors_per_dataset": int(frame["n_errors"].max()),
        "minimum_spi_validity_rate": float(min(rates.values())),
    }
    (args.data_dir / "feature_scout_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
