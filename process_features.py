"""
Compute and cache SPI-SPI feature matrices for downstream analysis.

- Loads datasets under a provided data path (e.g., data/full), enforcing consistent SPI ordering/flags.
- Directed SPIs are split into two pseudo-SPIs (i->j upper triangle, j->i lower triangle).
- Features are Spearman correlations between pseudo-SPI edge vectors (upper triangle of the SPI-SPI corr matrix).
- Supports optional SPI name subset via --spi-subset (txt, one per line).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning

from src.utils import load_json, project_root

import warnings

warnings.simplefilter("ignore", ConstantInputWarning)
LOGGER = logging.getLogger(__name__)


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


def _edge_vectors(name: str, mat: np.ndarray, directed: bool) -> List[tuple[str, np.ndarray]]:
    mat = np.asarray(mat, float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} is not square (shape={mat.shape})")
    if not directed:
        mat = 0.5 * (mat + mat.T)
        mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
        return [(name, _safe_zscore(mat[mask]))]
    upper_mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
    lower_mask = np.tril(np.ones(mat.shape, dtype=bool), k=-1)
    return [
        (f"{name}__ij", _safe_zscore(mat[upper_mask])),
        (f"{name}__ji", _safe_zscore(mat[lower_mask])),
    ]


def build_spi_spi_features(sample: Dict, spi_order: List[str], directed_flags: List[bool]) -> tuple[np.ndarray, List[str]]:
    vectors: List[np.ndarray] = []
    names: List[str] = []
    for name, directed in zip(spi_order, directed_flags):
        entries = _edge_vectors(name, sample["mpis"][name], directed)
        for pseudo_name, vec in entries:
            names.append(pseudo_name)
            vectors.append(vec)
    n = len(vectors)
    corr = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            r = spearmanr(vectors[i], vectors[j]).correlation
            corr[i, j] = corr[j, i] = 0.0 if not np.isfinite(r) else r
    iu = np.triu_indices(n, k=1)
    pairs = [(names[i], names[j]) for i, j in zip(iu[0], iu[1])]
    return corr[iu], names


def build_feature_matrix(samples: List[Dict], spi_order: List[str], directed_flags: List[bool]) -> tuple[np.ndarray, np.ndarray, List[tuple[str, str]], List[str]]:
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
        feat_vec, names = build_spi_spi_features(sample, spi_order, directed_flags)
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


def cache_path(data_path: str, limit: int | None, subset_label: str | None) -> Path:
    suffix = f"_limit{limit}" if limit else ""
    subset_suffix = f"_{subset_label}" if subset_label else ""
    safe = data_path.replace("\\", "-").replace("/", "-").strip("-")
    return project_root() / "analysis" / "feature_cache" / f"features_{safe}{subset_suffix}{suffix}.npz"


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
    parser.add_argument("--output", type=str, default=None, help="Output npz path (default: analysis/feature_cache/features_{data-path}[...] .npz)")
    parser.add_argument("--spi-subset", type=str, default=None, help="Path to txt file listing SPI names (one per line).")
    parser.add_argument("--recompute", action="store_true", help="Recompute even if cache exists.")
    parser.add_argument(
        "--mts-classes",
        type=str,
        default=None,
        help="Comma-separated mts_class names to include (filters class folders under the data path).",
    )
    return parser.parse_args(argv)


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

    cache_file = Path(args.output) if args.output else cache_path(args.data_path, args.dataset_limit, subset_label)
    cached = load_cached_features(cache_file, recompute=args.recompute)

    if cached:
        LOGGER.info("Cache exists, skipping computation: %s", cache_file)
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
    X_raw, y, pairs, dataset_paths, variants, Ms, Ts, instances = build_feature_matrix(samples, spi_order, directed_flags)
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
    }
    save_cached_features(cache_file, payload)
    LOGGER.info("Saved features -> %s", cache_file)


if __name__ == "__main__":
    main()
