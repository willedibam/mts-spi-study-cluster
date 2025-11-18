from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import ensure_dir, project_root, timestamp

LOGGER = logging.getLogger(__name__)


def _mode_results_dir(mode: str) -> Path:
    base = ensure_dir(project_root() / "results")
    return ensure_dir(base / mode)


def _load_feature_archive(mode: str) -> dict[str, np.ndarray]:
    path = _mode_results_dir(mode) / f"spi_space_features_{mode}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature archive {path} is missing. Run spi_space_features first."
        )
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _load_corr_matrices(dataset_paths: Sequence[str]) -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for rel_path in dataset_paths:
        arrays_dir = project_root() / rel_path / "arrays"
        corr_path = arrays_dir / "spi_spi_corr.npy"
        if not corr_path.exists():
            raise FileNotFoundError(
                f"Missing spi_spi_corr.npy at {corr_path}. Re-run spi_space_features."
            )
        matrices.append(np.load(corr_path))
    return matrices


def build_spi_subset_features(
    matrices: Sequence[np.ndarray],
    subset: Sequence[int],
) -> np.ndarray:
    subset = sorted(subset)
    if len(subset) < 2:
        return np.zeros((len(matrices), 0), dtype=np.float32)
    upper_idx = np.triu_indices(len(subset), k=1)
    features = np.empty((len(matrices), len(upper_idx[0])), dtype=np.float32)
    for row, matrix in enumerate(matrices):
        sub = matrix[np.ix_(subset, subset)]
        features[row] = sub[upper_idx]
    return features


def _make_classifier(name: str) -> Pipeline | RandomForestClassifier:
    if name == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        multi_class="auto",
                        C=1.0,
                        penalty="l2",
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported classifier '{name}'")


def _baseline_accuracy(y: np.ndarray, folds: Iterable[tuple[np.ndarray, np.ndarray]]) -> float:
    scores = []
    for train_idx, test_idx in folds:
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        train_labels = y[train_idx]
        unique, counts = np.unique(train_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        acc = float(np.mean(y[test_idx] == majority_label))
        scores.append(acc)
    return float(np.mean(scores)) if scores else 0.0


def _evaluate_subset(
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    if X.shape[1] == 0:
        return _baseline_accuracy(y, folds)
    if np.unique(y).size < 2:
        return 1.0
    base_clf = _make_classifier(classifier_name)
    scores: list[float] = []
    for train_idx, test_idx in folds:
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        train_labels = y[train_idx]
        if np.unique(train_labels).size < 2:
            scores.append(float(np.mean(y[test_idx] == train_labels[0])))
            continue
        clf = clone(base_clf)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        scores.append(float(accuracy_score(y[test_idx], preds)))
    return float(np.mean(scores)) if scores else 0.0


def _parse_classifiers(raw: str) -> list[str]:
    tokens = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]
    if not tokens:
        return ["logreg"]
    if "all" in tokens:
        return ["logreg", "rf"]
    valid = []
    for token in tokens:
        if token not in {"logreg", "rf"}:
            raise ValueError(f"Unsupported classifier '{token}'")
        if token not in valid:
            valid.append(token)
    return valid


def greedy_forward_selection(
    matrices: Sequence[np.ndarray],
    y: np.ndarray,
    *,
    spi_names: Sequence[str],
    max_spis: int,
    classifiers: Sequence[str],
    cv_folds: int,
) -> list[dict]:
    n_spis = len(spi_names)
    n_samples = len(y)
    if n_samples == 0:
        raise ValueError("No datasets available for selection.")
    if n_samples < 2:
        folds = [(np.arange(n_samples), np.arange(n_samples))]
    else:
        n_splits = min(cv_folds, n_samples)
        if n_splits < 2:
            n_splits = 2
        dummy = np.zeros((n_samples, 1))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = list(cv.split(dummy, y))
    all_results: list[dict] = []
    for clf_name in classifiers:
        selected: list[int] = []
        remaining = list(range(n_spis))
        scores: list[float] = []
        for _ in range(min(max_spis, n_spis)):
            best_idx = None
            best_score = -np.inf
            for candidate in remaining:
                subset = selected + [candidate]
                X_subset = build_spi_subset_features(matrices, subset)
                score = _evaluate_subset(X_subset, y, clf_name, folds)
                if score > best_score:
                    best_idx = candidate
                    best_score = score
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
            scores.append(best_score)
            LOGGER.info(
                "[%s] selected %d SPIs -> add %s (idx=%d) score=%.4f",
                clf_name,
                len(selected),
                spi_names[best_idx],
                best_idx,
                best_score,
            )
        all_results.append(
            {
                "classifier": clf_name,
                "selected_spi_indices": selected,
                "selected_spi_names": [spi_names[idx] for idx in selected],
                "scores": scores,
            }
        )
    return all_results


def run_pipeline(args: argparse.Namespace) -> Path:
    results_dir = _mode_results_dir(args.mode)
    archive = _load_feature_archive(args.mode)
    if args.label_type not in archive:
        raise KeyError(
            f"Label '{args.label_type}' not found in feature archive. "
            f"Available keys: {list(archive.keys())}"
        )
    labels = np.asarray(archive[args.label_type])
    if np.unique(labels).size < 2:
        LOGGER.warning(
            "Label '%s' has a single class across the selected datasets; "
            "scores will reflect this limitation.",
            args.label_type,
        )
    dataset_paths = archive["dataset_paths"].tolist()
    spi_names = archive["spi_names"].tolist()
    if args.limit:
        labels = labels[: args.limit]
        dataset_paths = dataset_paths[: args.limit]
    matrices = _load_corr_matrices(dataset_paths)
    if len(matrices) != len(labels):
        raise RuntimeError("Mismatch between matrices and labels.")
    LOGGER.info("Loaded %d datasets with %d SPIs each.", len(matrices), len(spi_names))
    classifiers = _parse_classifiers(args.classifier)
    selection = greedy_forward_selection(
        matrices,
        labels,
        spi_names=spi_names,
        max_spis=args.max_spis,
        classifiers=classifiers,
        cv_folds=args.cv_folds,
    )
    output_path = Path(
        args.output or (results_dir / f"spi_selection_{args.mode}.json")
    )
    ensure_dir(output_path.parent)
    payload = {
        "mode": args.mode,
        "label_type": args.label_type,
        "max_spis": args.max_spis,
        "cv_folds": args.cv_folds,
        "timestamp": timestamp(),
        "classifiers": selection,
        "spi_names": spi_names,
    }
    with Path(output_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOGGER.info("Saved SPI selection summary to %s", output_path)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Greedy forward selection over SPIs using SPI–SPI similarity matrices."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["dev", "full"],
        help="Dataset mode to analyse.",
    )
    parser.add_argument(
        "--label-type",
        default="mts_class",
        help="Label key from the feature archive (default: mts_class).",
    )
    parser.add_argument(
        "--max-spis",
        type=int,
        default=8,
        help="Maximum number of SPIs to select (default: 8).",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="logreg",
        help="Comma-separated list of classifiers (logreg, rf, all).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Stratified K-fold splits (default: 5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional dataset limit for debugging.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the selection JSON file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_pipeline(args)


if __name__ == "__main__":
    main()
