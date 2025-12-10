"""
SPI-SPI feature selection pipeline (cluster-friendly, no plots).

- Assumes features computed via feature_compute.py (same directed splitting).
- Loads cached SPI-SPI matrix if present, otherwise computes on the fly.
- Runs forward SequentialFeatureSelector with a configurable estimator.
- Writes selected features to analysis/feature_selection/<estimator>_<mode>_<k>_<timestamp>.txt
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from process_features import (
    build_feature_matrix,
    cache_path,
    load_cached_features,
    load_samples_with_flags,
    save_cached_features,
)
from src.utils import project_root

LOGGER = logging.getLogger(__name__)


# ----------------------------
# Estimators and selection
# ----------------------------
def make_estimator(name: str, n_jobs: int | None) -> object:
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=0,
            n_jobs=n_jobs if n_jobs and n_jobs > 0 else None,
        )
    if name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            n_jobs=n_jobs if n_jobs and n_jobs > 0 else None,
            solver="lbfgs",
            multi_class="auto",
        )
    if name == "linear_svc":
        return LinearSVC(
            dual=False,
            max_iter=5000,
        )
    raise ValueError(f"Unknown estimator '{name}'")


def run_selection(
    X_raw: np.ndarray,
    y: np.ndarray,
    *,
    target_dim: int,
    estimator_name: str,
    splits: int,
    n_jobs: int | None,
) -> dict:
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)
    target = min(target_dim, X_std.shape[1])
    min_class = int(pd.Series(y).value_counts().min())
    cv_splits = min(splits, min_class) if min_class >= 2 else 2
    est = make_estimator(estimator_name, n_jobs)
    sfs = SequentialFeatureSelector(
        est,
        n_features_to_select=target,
        direction="forward",
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0),
        n_jobs=n_jobs if n_jobs and n_jobs > 0 else None,
    )
    X_sel = sfs.fit_transform(X_std, y)
    support = sfs.get_support(indices=True)
    return {
        "X_selected": X_sel,
        "selected_indices": support,
        "cv_splits": cv_splits,
    }


# ----------------------------
# IO helpers
# ----------------------------
def save_selection_txt(
    path: Path,
    *,
    estimator: str,
    mode: str,
    target_dim: int,
    selected_pairs: List[tuple[str, str]],
    indices: np.ndarray,
    cv_splits: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# timestamp: {ts}\n")
        handle.write(f"# mode: {mode}\n")
        handle.write(f"# estimator: {estimator}\n")
        handle.write(f"# target_dim: {target_dim}\n")
        handle.write(f"# cv_splits: {cv_splits}\n")
        handle.write(f"# selected_features: {len(indices)}\n")
        handle.write("# index\tspi_a\tspi_b\n")
        for idx, feat_idx in enumerate(indices):
            pair = selected_pairs[feat_idx]
            handle.write(f"{idx}\t{pair[0]}\t{pair[1]}\n")


def output_path(estimator: str, mode: str, k: int) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{estimator}_{mode}_{k}_{stamp}.txt"
    return project_root() / "analysis" / "feature_selection" / fname


# ----------------------------
# Main
# ----------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPI-SPI forward feature selection (no plots).")
    parser.add_argument("--mode", default="full", help="Dataset mode (dev/full).")
    parser.add_argument("--dataset-limit", type=int, default=None, help="Optional dataset limit for quick runs.")
    parser.add_argument("--target-dim", type=int, default=12, help="Target feature count after selection.")
    parser.add_argument("--estimator", default="rf", choices=["rf", "logreg", "linear_svc"], help="Estimator used in SequentialFeatureSelector. {'rf', 'logreg', 'linear_svc'}")
    parser.add_argument("--splits", type=int, default=5, help="Max CV folds for selection.")
    parser.add_argument("--cache", type=str, default=None, help="Custom cache path (npz).")
    parser.add_argument("--recompute", action="store_true", help="Recompute feature matrix even if cached.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallelism for estimator/SFS (set thoughtfully on laptop).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cache_file = Path(args.cache) if args.cache else cache_path(args.mode, args.dataset_limit, None)
    cached = load_cached_features(cache_file, recompute=args.recompute)

    if cached:
        LOGGER.info("Loading features from cache: %s", cache_file)
        X_raw = cached["X"]
        y = cached["y"]
        pairs = list(cached["pairs"])
    else:
        LOGGER.info("Computing features from data/%s (limit=%s)", args.mode, args.dataset_limit)
        samples, spi_order, directed_flags = load_samples_with_flags(args.mode, limit=args.dataset_limit)
        X_raw, y, pairs, dataset_paths = build_feature_matrix(samples, spi_order, directed_flags)
        payload = {
            "X": X_raw.astype(np.float32),
            "y": y,
            "pairs": np.array(pairs, dtype=object),
            "spi_order": np.array(spi_order, dtype=object),
            "directed_flags": np.array(directed_flags, dtype=bool),
            "dataset_paths": np.array(dataset_paths, dtype=object),
        }
        save_cached_features(cache_file, payload)

    LOGGER.info("Feature matrix shape: %s x %s", X_raw.shape[0], X_raw.shape[1])
    if args.target_dim < 1:
        raise ValueError("target-dim must be >= 1")
    result = run_selection(
        X_raw,
        y,
        target_dim=args.target_dim,
        estimator_name=args.estimator,
        splits=args.splits,
        n_jobs=args.n_jobs,
    )
    indices = result["selected_indices"]
    selected_pairs = [pairs[i] for i in indices]
    out_path = output_path(args.estimator, args.mode, len(indices))
    save_selection_txt(
        out_path,
        estimator=args.estimator,
        mode=args.mode,
        target_dim=args.target_dim,
        selected_pairs=selected_pairs,
        indices=indices,
        cv_splits=result["cv_splits"],
    )
    LOGGER.info("Selected %d features -> %s", len(indices), out_path)


if __name__ == "__main__":
    main()
