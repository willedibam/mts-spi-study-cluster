"""
Compute and cache SPI-SPI feature matrices for downstream analysis.

- Loads datasets under a provided data path (e.g., data/full), enforcing consistent SPI ordering/flags.
- Directed SPIs can be split into two pseudo-SPIs (i->j upper triangle, j->i lower triangle) via --split-directed.
- Without splitting, directed SPIs are symmetrized and treated as a single SPI.
- Features are pairwise similarities between pseudo-SPI edge vectors (upper triangle of the SPI-SPI matrix).
- Supports optional SPI name subset via --spi-subset (txt, one per line).
- Supports different metrics via --metric (comma-separated): spearman, pearson, mi (mutual information).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning

from .utils import load_json, project_root

import warnings

warnings.simplefilter("ignore", ConstantInputWarning)
LOGGER = logging.getLogger(__name__)

MetricType = Literal["spearman", "pearson", "mi"]


def load_spi_subset(path: str | Path) -> tuple[list[str], str]:
    subset_path = Path(path)
    if not subset_path.exists():
        raise FileNotFoundError(f"SPI subset file not found: {subset_path}")
    names: List[str] = []
    with subset_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            name = raw.strip()
            if not name or name.startswith("#"):
                continue
            if name not in names:
                names.append(name)
    if not names:
        raise ValueError(f"SPI subset file is empty: {subset_path}")
    return names, subset_path.name


def load_samples_with_flags(
    data_path: str | Path,
    limit: int | None = None,
    subset_names: List[str] | None = None,
    mts_classes: List[str] | None = None,
) -> tuple:
    samples: List[Dict] = []
    spi_order: List[str] | None = None
    directed_flags: List[bool] | None = None
    base = Path(data_path)
    if not base.exists():
        raise FileNotFoundError(f"Data path not found: {base}")
    class_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if mts_classes:
        allowed = set(mts_classes)
        class_dirs = [p for p in class_dirs if p.name in allowed]
    for class_dir in class_dirs:
        for ds_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            meta_path = ds_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = load_json(meta_path)
            spis = meta["pyspi"]["spis"]
            if subset_names:
                by_name = {s["name"]: s for s in spis}
                missing = [name for name in subset_names if name not in by_name]
                if missing:
                    raise ValueError(f"Dataset {ds_dir} missing SPI(s): {', '.join(missing)}")
                spis = [by_name[name] for name in subset_names]
            order = [e["name"] for e in spis]
            flags = [e.get("directed", False) for e in spis]
            if spi_order is None:
                spi_order, directed_flags = order, flags
            else:
                if order != spi_order:
                    raise ValueError(f"SPI order mismatch in {ds_dir}")
                if flags != directed_flags:
                    raise ValueError(f"Directed flags mismatch in {ds_dir}")
            with np.load(ds_dir / "spi_mpis.npz") as npz:
                mpis = {k: npz[k] for k in npz.files}
            samples.append(
                {
                    "label": meta["mts_class"],
                    "mpis": mpis,
                    "M": meta.get("M"),
                    "T": meta.get("T"),
                    "path": ds_dir,
                    "variant": (meta.get("variant") or {}).get("name", "") if isinstance(meta.get("variant"), dict) else (meta.get("variant") or ""),
                    "instance": meta.get("instance_index"),
                }
            )
            if limit and len(samples) >= limit:
                break
        if limit and len(samples) >= limit:
            break
    if spi_order is None or directed_flags is None:
        raise RuntimeError(f"No datasets found for data_path={data_path}")
    return samples, spi_order, directed_flags


def _safe_zscore(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, float)
    std = vec.std()
    if std < 1e-12 or not np.isfinite(std):
        return np.zeros_like(vec)
    return (vec - vec.mean()) / std


def _edge_vectors(
    name: str,
    mat: np.ndarray,
    directed: bool,
    split_directed: bool = False,
) -> List[tuple[str, np.ndarray]]:
    mat = np.asarray(mat, float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} is not square (shape={mat.shape})")
    if not directed:
        mat = 0.5 * (mat + mat.T)
        mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
        return [(name, _safe_zscore(mat[mask]))]
    if not split_directed:
        # Collapse direction to keep a single vector per SPI.
        mat = 0.5 * (mat + mat.T)
        mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
        return [(name, _safe_zscore(mat[mask]))]
    upper_mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
    lower_mask = np.tril(np.ones(mat.shape, dtype=bool), k=-1)
    return [
        (f"{name}__ij", _safe_zscore(mat[upper_mask])),
        (f"{name}__ji", _safe_zscore(mat[lower_mask])),
    ]


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Vectorized ranking along axis 1 (rows)."""
    n = arr.shape[1]
    order = np.argsort(arr, axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(arr.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, n + 1)
    return ranks


def _pearson_corr_matrix(V: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix for row vectors."""
    V = V - V.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    V = V / norms
    return V @ V.T


def _spearman_corr_matrix(V: np.ndarray) -> np.ndarray:
    """Compute Spearman correlation matrix (Pearson on ranks)."""
    R = _rankdata(V)
    return _pearson_corr_matrix(R)


def _mutual_information_matrix(V: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """
    Compute pairwise mutual information matrix using histogram-based estimation.
    Returns normalized MI (0 to 1 range).
    """
    n_vectors = V.shape[0]
    mi_matrix = np.zeros((n_vectors, n_vectors), dtype=np.float64)
    digitized = np.zeros_like(V, dtype=np.int32)
    for i in range(n_vectors):
        vec = V[i]
        vec_min, vec_max = vec.min(), vec.max()
        if vec_max - vec_min < 1e-12:
            digitized[i] = 0
        else:
            bins = np.linspace(vec_min, vec_max, n_bins + 1)
            digitized[i] = np.clip(np.digitize(vec, bins) - 1, 0, n_bins - 1)
    for i in range(n_vectors):
        mi_matrix[i, i] = 1.0  # Self MI normalized to 1
        for j in range(i + 1, n_vectors):
            joint_hist = np.zeros((n_bins, n_bins), dtype=np.float64)
            np.add.at(joint_hist, (digitized[i], digitized[j]), 1)
            joint_hist /= joint_hist.sum()
            p_i = joint_hist.sum(axis=1)
            p_j = joint_hist.sum(axis=0)
            outer = np.outer(p_i, p_j)
            mask = (joint_hist > 0) & (outer > 0)
            mi = np.sum(joint_hist[mask] * np.log(joint_hist[mask] / outer[mask]))
            h_i = -np.sum(p_i[p_i > 0] * np.log(p_i[p_i > 0]))
            h_j = -np.sum(p_j[p_j > 0] * np.log(p_j[p_j > 0]))
            min_h = min(h_i, h_j)
            if min_h > 1e-12:
                mi_normalized = mi / min_h
            else:
                mi_normalized = 0.0
            mi_matrix[i, j] = mi_matrix[j, i] = mi_normalized
    return mi_matrix


def build_spi_spi_features(
    sample: Dict,
    spi_order: List[str],
    directed_flags: List[bool],
    *,
    split_directed: bool = False,
    metric: MetricType = "spearman",
) -> tuple[np.ndarray, List[str]]:
    vectors: List[np.ndarray] = []
    names: List[str] = []
    for name, directed in zip(spi_order, directed_flags):
        entries = _edge_vectors(name, sample["mpis"][name], directed, split_directed)
        for pseudo_name, vec in entries:
            names.append(pseudo_name)
            vectors.append(vec)
            
    V = np.vstack(vectors).astype(np.float64)
    n = V.shape[0]
    
    if metric == "spearman":
        corr = _spearman_corr_matrix(V)
    elif metric == "pearson":
        corr = _pearson_corr_matrix(V)
    elif metric == "mi":
        corr = _mutual_information_matrix(V)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    corr = np.where(np.isfinite(corr), corr, 0.0).astype(np.float32)
    iu = np.triu_indices(n, k=1)
    return corr[iu], names


def build_feature_matrix(
    samples: List[Dict],
    spi_order: List[str],
    directed_flags: List[bool],
    *,
    split_directed: bool = False,
    metric: MetricType = "spearman",
) -> tuple[np.ndarray, np.ndarray, List[tuple[str, str]], List[str]]:
    X_list: List[np.ndarray] = []
    y_list: List[str] = []
    dataset_paths: List[str] = []
    variants: List[str] = []
    Ms: List[int] = []
    Ts: List[int] = []
    instances: List[int] = []
    names_ref: List[str] | None = None
    pairs: List[tuple[str, str]] | None = None
    for idx, sample in enumerate(samples, start=1):
        feat_vec, names = build_spi_spi_features(
            sample,
            spi_order,
            directed_flags,
            split_directed=split_directed,
            metric=metric,
        )
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise ValueError("Pseudo-SPI ordering mismatch across datasets.")
        if pairs is None:
            n = len(names)
            pairs = [(names[i], names[j]) for i in range(n) for j in range(i + 1, n)]
        X_list.append(feat_vec)
        y_list.append(sample["label"])
        dataset_paths.append(str(sample["path"]))
        variants.append(sample.get("variant", ""))
        Ms.append(sample.get("M"))
        Ts.append(sample.get("T"))
        instances.append(sample.get("instance"))
        if idx % 10 == 0 or idx == len(samples):
            LOGGER.info("features: %d/%d (%.0f%%)", idx, len(samples), 100 * idx / len(samples))
    if names_ref is None or pairs is None:
        raise RuntimeError("No features computed.")
    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y, pairs, dataset_paths, variants, Ms, Ts, instances


def cache_path(
    data_path: str,
    limit: int | None,
    subset_label: str | None,
    *,
    split_directed: bool = False,
    metric: MetricType = "spearman",
) -> Path:
    suffix = f"_limit{limit}" if limit else ""
    subset_suffix = f"_{subset_label}" if subset_label else ""
    split_suffix = "_split" if split_directed else ""
    metric_suffix = f"_{metric}"
    safe = data_path.replace("\\", "-").replace("/", "-").strip("-")
    return project_root() / "analysis" / "feature_cache" / f"{safe}{split_suffix}{metric_suffix}{subset_suffix}{suffix}.npz"


def load_cached_features(path: Path, recompute: bool) -> dict | None:
    if recompute or not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_cached_features(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    LOGGER.info("Cached features -> %s", path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and cache SPI-SPI feature matrices.")
    parser.add_argument("--data-path", default="data/full", help="Path to data root (e.g., data/full, data/full-variants).")
    parser.add_argument("--dataset-limit", type=int, default=None, help="Optional dataset limit for quick runs.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output npz path (if multiple metrics, suffixes are added per metric).",
    )
    parser.add_argument("--spi-subset", type=str, default=None, help="Path to txt file listing SPI names (one per line).")
    parser.add_argument("--recompute", action="store_true", help="Recompute even if cache exists.")
    parser.add_argument(
        "--split-directed",
        action="store_true",
        help="Split directed SPIs into two pseudo-SPIs (upper/lower). Default: off (symmetrize into one).",
    )
    parser.add_argument(
        "--mts-classes",
        type=str,
        default=None,
        help="Comma-separated mts_class names to include (filters class folders under the data path).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="spearman",
        help="Comma-separated metrics to compute: spearman (default), pearson, mi.",
    )
    return parser.parse_args(argv)


def _parse_metrics(raw: str | None) -> List[MetricType]:
    if not raw:
        return ["spearman"]
    metrics = [m.strip().lower() for m in raw.split(",") if m.strip()]
    if not metrics:
        return ["spearman"]
    allowed = {"spearman", "pearson", "mi"}
    unknown = [m for m in metrics if m not in allowed]
    if unknown:
        raise ValueError(f"Unknown metric(s): {', '.join(sorted(set(unknown)))}")
    seen = set()
    ordered: List[MetricType] = []
    for m in metrics:
        if m not in seen:
            ordered.append(m)  # type: ignore[arg-type]
            seen.add(m)
    return ordered


def _output_path_for_metric(base: str, metric: MetricType, multi: bool) -> Path:
    out = Path(base)
    if not multi:
        return out
    suffix = out.suffix
    stem = out.stem if suffix else out.name
    return out.with_name(f"{stem}_{metric}{suffix}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    subset_names: List[str] | None = None
    subset_label: str | None = None
    if args.spi_subset:
        subset_names, subset_label = load_spi_subset(args.spi_subset)
        LOGGER.info("Using SPI subset: %s (%d SPIs)", subset_label, len(subset_names))
    else:
        LOGGER.info("Using all SPIs")
    
    metrics = _parse_metrics(args.metric)
    multi_metric = len(metrics) > 1
    LOGGER.info("Using metric(s): %s", ", ".join(metrics))

    targets: List[tuple[MetricType, Path]] = []
    for metric in metrics:
        cache_file = (
            _output_path_for_metric(args.output, metric, multi_metric)
            if args.output
            else cache_path(
                args.data_path,
                args.dataset_limit,
                subset_label,
                split_directed=args.split_directed,
                metric=metric,
            )
        )
        cached = load_cached_features(cache_file, recompute=args.recompute)
        if cached:
            LOGGER.info("Cache exists, skipping computation: %s", cache_file)
        else:
            targets.append((metric, cache_file))

    if not targets:
        return

    mts_classes = (
        [part.strip() for part in args.mts_classes.split(",") if part.strip()]
        if args.mts_classes
        else None
    )
    samples, spi_order, directed_flags = load_samples_with_flags(
        args.data_path,
        limit=args.dataset_limit,
        subset_names=subset_names,
        mts_classes=mts_classes,
    )
    for metric, cache_file in targets:
        LOGGER.info("Computing features (metric=%s)", metric)
        X_raw, y, pairs, dataset_paths, variants, Ms, Ts, instances = build_feature_matrix(
            samples,
            spi_order,
            directed_flags,
            split_directed=args.split_directed,
            metric=metric,
        )
        payload = {
            "X": X_raw.astype(np.float32),
            "y": y,  # mts_class labels
            "pairs": np.array(pairs, dtype=object),
            "spi_order": np.array(spi_order, dtype=object),
            "directed_flags": np.array(directed_flags, dtype=bool),
            "dataset_paths": np.array(dataset_paths, dtype=object),
            "variant": np.array(variants, dtype=object),
            "M": np.array(Ms, dtype=object),
            "T": np.array(Ts, dtype=object),
            "instance": np.array(instances, dtype=object),
            "mode": args.data_path,
            "dataset_limit": args.dataset_limit if args.dataset_limit is not None else -1,
            "spi_subset": subset_label or "",
            "split_directed": bool(args.split_directed),
            "metric": metric,
        }
        save_cached_features(cache_file, payload)
        LOGGER.info("Saved features -> %s", cache_file)


if __name__ == "__main__":
    main()
