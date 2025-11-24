from __future__ import annotations

import argparse
import json
import logging
import warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import f_oneway, ks_2samp, ttest_ind
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .plot_style import apply_plot_style
from .utils import ensure_dir, project_root, slugify, timestamp

LOGGER = logging.getLogger(__name__)

METHOD_CHOICES = ("greedy", "ks", "ttest", "fstat", "all")
STAT_METHODS = {"ks", "ttest", "fstat"}
STAT_MODES = ("pairwise", "ovr")

def _results_dir(mode: str, space: str, category: str) -> Path:
    base = ensure_dir(project_root() / "results" / mode / space)
    return ensure_dir(base / category)


def _load_feature_archive(mode: str, space: str) -> dict[str, np.ndarray]:
    path = _results_dir(mode, space, "features") / f"spi_space_features_{mode}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Feature archive {path} is missing. Run 'python -m src.features --mode {mode} --space {space}' first."
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
                f"Missing spi_spi_corr.npy at {corr_path}. Re-run 'python -m src.features'."
            )
        matrices.append(np.load(corr_path))
    return matrices


def build_spi_subset_features(
    matrices: Sequence[np.ndarray],
    subset: Sequence[int],
    *,
    feature_space: str,
    n_spis: int,
) -> np.ndarray:
    subset = sorted(subset)
    if feature_space == "spi-spi":
        if len(subset) < 2:
            return np.zeros((len(matrices), 0), dtype=np.float32)
        upper_idx = np.triu_indices(len(subset), k=1)
        features = np.empty((len(matrices), len(upper_idx[0])), dtype=np.float32)
        for row, matrix in enumerate(matrices):
            sub = matrix[np.ix_(subset, subset)]
            features[row] = sub[upper_idx]
        return features
    if feature_space == "spi":
        if not subset:
            return np.zeros((len(matrices), 0), dtype=np.float32)
        if n_spis < 2:
            raise ValueError(
                "Need at least two SPIs recorded to build SPI-level features."
            )
        per_spi_length = n_spis - 1
        features = np.empty(
            (len(matrices), len(subset) * per_spi_length), dtype=np.float32
        )
        for row, matrix in enumerate(matrices):
            row_features = []
            for spi_idx in subset:
                row_vals = np.concatenate(
                    (matrix[spi_idx, :spi_idx], matrix[spi_idx, spi_idx + 1 :])
                )
                row_features.append(row_vals)
            features[row] = np.concatenate(row_features)
        return features
    raise ValueError(f"Unsupported feature_space '{feature_space}'")


def _make_classifier(name: str) -> Pipeline | RandomForestClassifier:
    if name == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
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


def _build_folds(labels: np.ndarray, cv_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if labels.size < 2 or np.unique(labels).size < 2:
        idx = np.arange(labels.size)
        return [(idx, idx)]
    n_splits = min(cv_folds, labels.size)
    if n_splits < 2:
        n_splits = 2
    dummy = np.zeros((labels.size, 1))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return list(splitter.split(dummy, labels))


def _evaluate_order_accuracy(
    order: list[int],
    matrices: Sequence[np.ndarray],
    labels: np.ndarray,
    feature_space: str,
    n_spis: int,
    classifier_name: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    scores: list[float] = []
    selected: list[int] = []
    for idx in order:
        selected.append(idx)
        X_subset = build_spi_subset_features(
            matrices,
            selected,
            feature_space=feature_space,
            n_spis=n_spis,
        )
        score = _evaluate_subset(X_subset, labels, classifier_name, folds)
        scores.append(score)
    return scores


def _plot_accuracy_curves(curves: dict[str, list[float]], path: Path, title: str) -> None:
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for label, values in curves.items():
        if not values:
            continue
        ax.plot(
            range(1, len(values) + 1),
            values,
            marker="o",
            label=label,
        )
    ax.set_xlabel("Number of SPIs")
    ax.set_ylabel("CV accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_stat_scores(
    spi_names: Sequence[str],
    indices: list[int],
    scores: list[float],
    path: Path,
    title: str,
) -> None:
    if not indices:
        return
    labels = [spi_names[i] for i in indices]
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(indices)))
    ax.bar(range(len(indices)), scores, color=colors)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Statistic score")
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _compute_spi_means(X: np.ndarray, spi_slices: np.ndarray) -> np.ndarray:
    n_datasets = X.shape[0]
    n_spis = spi_slices.shape[0]
    means = np.zeros((n_datasets, n_spis), dtype=np.float32)
    for idx in range(n_spis):
        start, length = spi_slices[idx]
        start = int(start)
        length = int(length)
        block = X[:, start : start + length]
        means[:, idx] = block.mean(axis=1)
    return means


def _base_color_map(base_classes: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    ordered = []
    for cls in base_classes:
        if cls not in ordered:
            ordered.append(cls)
    palette = sns.color_palette("pastel", n_colors=max(1, len(ordered)))
    return {cls: palette[i] for i, cls in enumerate(ordered)}


def _label_color_map(
    labels: Sequence[str],
    base_classes: Sequence[str],
    base_colors: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float]]:
    colors: dict[str, tuple[float, float, float]] = {}
    label_list = list(labels)
    base_list = list(base_classes)
    for label in dict.fromkeys(label_list):
        indices = [i for i, lbl in enumerate(label_list) if lbl == label]
        if not indices:
            continue
        bases = [base_list[i] for i in indices]
        dominant = max(set(bases), key=bases.count)
        colors[label] = base_colors.get(dominant, (0.6, 0.6, 0.6))
    return colors


def _plot_violin(
    values: np.ndarray,
    labels: np.ndarray,
    selected: list[str],
    colors: dict[str, tuple[float, float, float]],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    if not selected:
        return
    apply_plot_style()
    label_order_global = list(dict.fromkeys(labels.tolist()))
    plt.rcParams["text.usetex"] = False
    mask = np.isin(labels, selected)
    if not mask.any():
        return
    data_vals = values[mask]
    data_labels = labels[mask]
    order = [lbl for lbl in selected if lbl in set(data_labels)]
    if not order:
        order = [lbl for lbl in label_order_global if lbl in set(data_labels)]
    if len(order) < 2 and len(set(data_labels)) < 2:
        return
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    palette_dict = {lbl: colors.get(lbl, (0.6, 0.6, 0.6)) for lbl in order}
    df = pd.DataFrame({"label": data_labels, "value": data_vals})
    sns.violinplot(
        data=df,
        x="label",
        y="value",
        hue="label",
        order=order,
        hue_order=order,
        palette=palette_dict,
        cut=0,
        legend=False,
        dodge=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha("right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _statistic_between(
    block: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    method: str,
) -> float:
    arr_a = block[mask_a]
    arr_b = block[mask_b]
    if arr_a.size == 0 or arr_b.size == 0:
        return 0.0
    stats_values: list[float] = []
    for col in range(block.shape[1]):
        series_a = arr_a[:, col]
        series_b = arr_b[:, col]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "ks":
                stat = ks_2samp(series_a, series_b, mode="auto").statistic
            elif method == "ttest":
                stat = np.abs(
                    ttest_ind(series_a, series_b, equal_var=False, nan_policy="omit").statistic
                )
            elif method == "fstat":
                stat = f_oneway(series_a, series_b).statistic
            else:
                stat = 0.0
        stats_values.append(
            float(np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0))
        )
    return max(stats_values) if stats_values else 0.0


def _compute_stat_matrices(
    X: np.ndarray,
    spi_slices: np.ndarray,
    labels: np.ndarray,
    method: str,
) -> tuple[np.ndarray, list[tuple[str, str]], np.ndarray, np.ndarray]:
    n_spis = spi_slices.shape[0]
    classes = np.unique(labels)
    combos = list(combinations(classes, 2))
    pair_matrix = np.zeros((n_spis, len(combos)), dtype=float)
    ovr_matrix = np.zeros((n_spis, len(classes)), dtype=float)
    for idx in range(n_spis):
        start, length = spi_slices[idx]
        start = int(start)
        length = int(length)
        block = X[:, start : start + length]
        for pair_idx, (a, b) in enumerate(combos):
            mask_a = labels == a
            mask_b = labels == b
            pair_matrix[idx, pair_idx] = _statistic_between(
                block, mask_a, mask_b, method
            )
        for cls_idx, cls in enumerate(classes):
            mask_a = labels == cls
            mask_b = labels != cls
            ovr_matrix[idx, cls_idx] = _statistic_between(block, mask_a, mask_b, method)
    return pair_matrix, combos, ovr_matrix, classes


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
    feature_space: str,
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
                X_subset = build_spi_subset_features(
                    matrices,
                    subset,
                    feature_space=feature_space,
                    n_spis=n_spis,
                )
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
            if len(selected) >= max_spis or best_score >= 0.9999:
                LOGGER.info(
                    "[%s] stopping early after reaching score %.4f",
                    clf_name,
                    best_score,
                )
                break
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
    features_dir = _results_dir(args.mode, args.feature_space, "features")
    archive = _load_feature_archive(args.mode, args.feature_space)
    space_in_archive = str(archive.get("space") or args.feature_space)
    if space_in_archive != args.feature_space:
        raise ValueError(
            f"Archive was generated for space '{space_in_archive}', but '--feature-space {args.feature_space}' was requested."
        )
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
    display_classes = np.array(
        archive.get("display_class", archive["dataset_paths"]), dtype=object
    )
    base_classes = np.array(
        archive.get("base_class", archive[args.label_type]), dtype=object
    )
    X_archive = np.array(archive["X"], dtype=float)
    if args.limit:
        labels = labels[: args.limit]
        dataset_paths = dataset_paths[: args.limit]
        X_archive = X_archive[: args.limit]
        display_classes = display_classes[: args.limit]
        base_classes = base_classes[: args.limit]
    matrices = _load_corr_matrices(dataset_paths)
    if len(matrices) != len(labels):
        raise RuntimeError("Mismatch between matrices and labels.")
    LOGGER.info("Loaded %d datasets with %d SPIs each.", len(matrices), len(spi_names))
    n_spis = len(spi_names)

    methods = (
        ["greedy", "ks", "ttest", "fstat"]
        if args.method == "all"
        else [args.method]
    )
    need_stats = any(m in STAT_METHODS for m in methods)
    spi_means = None
    spi_feature_slices = None
    label_colors: dict[str, tuple[float, float, float]] = {}
    X_spi = None
    if need_stats:
        if args.feature_space == "spi":
            spi_archive = archive
        else:
            spi_archive = _load_feature_archive(args.mode, "spi")
        X_spi = np.array(spi_archive["X"], dtype=float)
        spi_feature_slices = spi_archive.get("spi_feature_slices")
        if spi_feature_slices is None or spi_feature_slices.size == 0:
            raise ValueError(
                "SPI feature slices missing from SPI archive; recompute features with --space spi."
            )
        if args.limit:
            X_spi = X_spi[: args.limit]
        spi_means = _compute_spi_means(X_spi, spi_feature_slices)
        base_palette = _base_color_map(base_classes)
        label_colors = _label_color_map(labels, base_classes, base_palette)
    last_output: Path | None = None
    for method in methods:
        method_dir = ensure_dir(features_dir / method)
        plots_dir = ensure_dir(method_dir / "plots")
        if method == "greedy":
            classifiers = _parse_classifiers(args.classifier)
            selection = greedy_forward_selection(
                matrices,
                labels,
                spi_names=spi_names,
                max_spis=args.max_spis,
                classifiers=classifiers,
                cv_folds=args.cv_folds,
                feature_space=args.feature_space,
            )
            best_entry = max(
                selection,
                key=lambda entry: entry["scores"][-1] if entry["scores"] else -np.inf,
                default=None,
            )
            payload = {
                "mode": args.mode,
                "label_type": args.label_type,
                "max_spis": args.max_spis,
                "cv_folds": args.cv_folds,
                "feature_space": args.feature_space,
                "timestamp": timestamp(),
                "method": "greedy",
                "classifiers": selection,
                "spi_names": spi_names,
            }
            if best_entry:
                payload["selected_spi_indices"] = best_entry["selected_spi_indices"]
                payload["selected_spi_names"] = best_entry["selected_spi_names"]
                payload["scores"] = best_entry["scores"]
            output_path = Path(
                args.output
                or (
                    method_dir
                    / f"greedy_{args.mode}_{args.feature_space}_{args.label_type}.json"
                )
            )
            ensure_dir(output_path.parent)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            LOGGER.info("Saved greedy selection to %s", output_path)
            curves = {entry["classifier"]: entry["scores"] for entry in selection}
            _plot_accuracy_curves(
                curves,
                plots_dir / f"greedy_{args.mode}_{args.feature_space}_accuracy.png",
                "Greedy forward selection",
            )
            last_output = output_path
            continue

        if spi_feature_slices is None or spi_means is None:
            raise RuntimeError("SPI data unavailable for statistical selectors.")
        pair_matrix, combos, ovr_matrix, class_labels = _compute_stat_matrices(
            X_spi,
            spi_feature_slices,
            labels,
            method=method,
        )
        pair_scores = pair_matrix.max(axis=1) if pair_matrix.size else np.zeros(n_spis)
        ovr_scores = ovr_matrix.max(axis=1) if ovr_matrix.size else np.zeros(n_spis)
        if args.stat_mode == "pairwise":
            ranking = np.argsort(pair_scores)[::-1]
            stat_scores = pair_scores
        else:
            ranking = np.argsort(ovr_scores)[::-1]
            stat_scores = ovr_scores
        selected_order = ranking[: min(args.max_spis, len(spi_names))].tolist()
        stat_classifier = _parse_classifiers(args.classifier)[0]
        folds = _build_folds(labels, args.cv_folds)
        accuracy_curve = _evaluate_order_accuracy(
            selected_order,
            matrices,
            labels,
            feature_space=args.feature_space,
            n_spis=n_spis,
            classifier_name=stat_classifier,
            folds=folds,
        )
        payload = {
            "mode": args.mode,
            "label_type": args.label_type,
            "feature_space": args.feature_space,
            "timestamp": timestamp(),
            "method": method,
            "stat_mode": args.stat_mode,
            "selected_spi_indices": selected_order,
            "selected_spi_names": [spi_names[i] for i in selected_order],
            "stat_scores": [float(stat_scores[i]) for i in selected_order],
            "cv_scores": accuracy_curve,
            "classifier": stat_classifier,
            "spi_names": spi_names,
        }
        summary_pair: dict[str, list[dict[str, object]]] = {}
        summary_ovr: dict[str, list[dict[str, object]]] = {}
        pair_base = method_dir / "pairwise"
        ovr_base = method_dir / "ovr"
        top_k = min(10, n_spis)
        for pair_idx, (cls_a, cls_b) in enumerate(combos):
            stats = pair_matrix[:, pair_idx]
            order = np.argsort(stats)[::-1][:top_k]
            key = f"{cls_a}__vs__{cls_b}"
            summary_pair[key] = [
                {
                    "spi_index": int(idx),
                    "spi_name": spi_names[idx],
                    "statistic": float(stats[idx]),
                }
                for idx in order
            ]
            pair_dir = pair_base / slugify(key, fallback="pair")
            violins_dir = pair_dir / "violins"
            ensure_dir(violins_dir)
            with (pair_dir / "top_features.json").open("w", encoding="utf-8") as handle:
                json.dump(summary_pair[key], handle, indent=2)
            for idx in order:
                values = spi_means[:, idx]
                filename = violins_dir / f"{slugify(spi_names[idx])}.png"
                _plot_violin(
                    values,
                    labels,
                    [cls_a, cls_b],
                    label_colors,
                    spi_names[idx],
                    "Mean Feature Value",
                    filename,
                )
        for cls_idx, cls in enumerate(class_labels):
            stats = ovr_matrix[:, cls_idx]
            order = np.argsort(stats)[::-1][:top_k]
            summary_ovr[str(cls)] = [
                {
                    "spi_index": int(idx),
                    "spi_name": spi_names[idx],
                    "statistic": float(stats[idx]),
                }
                for idx in order
            ]
            class_dir = ovr_base / slugify(str(cls), fallback="class")
            violins_dir = class_dir / "violins"
            ensure_dir(violins_dir)
            with (class_dir / "top_features.json").open("w", encoding="utf-8") as handle:
                json.dump(summary_ovr[str(cls)], handle, indent=2)
            for idx in order:
                values = spi_means[:, idx]
                filename = violins_dir / f"{slugify(spi_names[idx])}.png"
                _plot_violin(
                    values,
                    labels,
                    list(np.unique(labels)),
                    label_colors,
                    spi_names[idx],
                    "Mean Feature Value",
                    filename,
                )
        payload["pairwise_top_features"] = summary_pair
        payload["ovr_top_features"] = summary_ovr
        filename = (
            args.output
            if args.method != "all" and args.output
            else method_dir
            / f"{method}_{args.stat_mode}_top{len(selected_order)}.json"
        )
        output_path = Path(filename)
        ensure_dir(output_path.parent)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        LOGGER.info("Saved %s selection to %s", method, output_path)
        _plot_accuracy_curves(
            {stat_classifier: accuracy_curve},
            plots_dir
            / f"{method}_{args.stat_mode}_{args.mode}_{args.feature_space}_accuracy.png",
            f"{method.upper()} accuracy curve",
        )
        _plot_stat_scores(
            spi_names,
            selected_order,
            [stat_scores[i] for i in selected_order],
            plots_dir
            / f"{method}_{args.stat_mode}_{args.mode}_{args.feature_space}_stats.png",
            f"{method.upper()} statistic ({args.stat_mode})",
        )
        last_output = output_path
    return last_output or Path(features_dir)


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
        "--method",
        choices=list(METHOD_CHOICES),
        default="greedy",
        help="Selection strategy to use (default: greedy).",
    )
    parser.add_argument(
        "--stat-mode",
        choices=list(STAT_MODES),
        default="pairwise",
        help="Statistic aggregation mode for KS/t/F selectors.",
    )
    parser.add_argument(
        "--feature-space",
        choices=["spi-spi", "spi"],
        default="spi",
        help="Use SPI–SPI submatrices ('spi-spi') or SPI-level rows ('spi') during evaluation.",
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
