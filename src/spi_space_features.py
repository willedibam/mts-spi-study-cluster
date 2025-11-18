from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr
from tqdm import tqdm

from .utils import ensure_dir, iter_dataset_dirs, load_json, project_root, to_relative

LOGGER = logging.getLogger(__name__)


def _mode_results_dir(mode: str) -> Path:
    base = ensure_dir(project_root() / "results")
    return ensure_dir(base / mode)


@dataclass
class DatasetRecord:
    dataset_dir: Path
    dataset_path: str
    mts_class: str
    variant: str
    M: int
    T: int
    instance_index: int
    spi_names: List[str]
    spi_directed: List[bool]
    per_spi_paths: dict[str, str]


@dataclass
class SPIVectors:
    name: str
    directed: bool
    upper: np.ndarray
    full: np.ndarray

    def vector_pair(self, other: "SPIVectors") -> tuple[np.ndarray, np.ndarray]:
        if not self.directed and not other.directed:
            return self.upper, other.upper
        return self.full, other.full


def _load_variant_name(meta: dict) -> str:
    variant = meta.get("variant")
    if isinstance(variant, dict):
        return variant.get("name") or ""
    if isinstance(variant, str):
        return variant
    return ""


def enumerate_datasets(mode: str) -> Iterator[DatasetRecord]:
    for dataset_dir in iter_dataset_dirs(mode):
        meta_path = dataset_dir / "meta.json"
        if not meta_path.exists():
            LOGGER.warning("Skipping %s because meta.json is missing.", dataset_dir)
            continue
        meta = load_json(meta_path)
        pyspi_meta = meta.get("pyspi") or {}
        spi_entries = pyspi_meta.get("spis") or []
        if not spi_entries:
            LOGGER.warning("Skipping %s because no SPI metadata was recorded.", dataset_dir)
            continue
        spi_names = [entry.get("name") for entry in spi_entries]
        if not all(isinstance(name, str) and name for name in spi_names):
            LOGGER.warning("Skipping %s because some SPI names are invalid.", dataset_dir)
            continue
        directed = [bool(entry.get("directed", False)) for entry in spi_entries]
        per_spi_paths = pyspi_meta.get("per_spi") or {}
        record = DatasetRecord(
            dataset_dir=dataset_dir,
            dataset_path=to_relative(dataset_dir),
            mts_class=meta.get("mts_class", meta.get("class", "unknown")),
            variant=_load_variant_name(meta),
            M=int(meta.get("M", 0)),
            T=int(meta.get("T", 0)),
            instance_index=int(meta.get("instance_index", -1)),
            spi_names=spi_names,
            spi_directed=directed,
            per_spi_paths=per_spi_paths,
        )
        yield record


def _load_spi_matrices(record: DatasetRecord) -> dict[str, np.ndarray]:
    dataset_dir = record.dataset_dir
    spi_names = record.spi_names
    matrices: dict[str, np.ndarray] = {}
    archive = dataset_dir / "spi_mpis.npz"
    if archive.exists():
        with np.load(archive) as npz:
            for name in spi_names:
                if name in npz:
                    matrices[name] = npz[name]
    missing = [name for name in spi_names if name not in matrices]
    if missing:
        arrays_dir = dataset_dir / "arrays"
        for name in missing:
            rel_path = record.per_spi_paths.get(name)
            if rel_path:
                cand = dataset_dir / rel_path
            else:
                safe_name = name.replace("/", "_")
                cand = arrays_dir / f"mpi_{safe_name}.npy"
            if not cand.exists():
                raise FileNotFoundError(
                    f"Missing MPI array for SPI '{name}' in {dataset_dir}"
                )
            matrices[name] = np.load(cand)
    return matrices


def _vectorise_spi_matrices(
    record: DatasetRecord,
    matrices: dict[str, np.ndarray],
    *,
    zscore: bool,
) -> list[SPIVectors]:
    vectors: list[SPIVectors] = []
    for name, directed in zip(record.spi_names, record.spi_directed):
        matrix = np.asarray(matrices[name], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"SPI '{name}' for {record.dataset_path} does not provide a square matrix."
            )
        M = matrix.shape[0]
        if not directed:
            # Ensure symmetry to avoid floating point mismatches
            matrix = 0.5 * (matrix + matrix.T)
        upper_idx = np.triu_indices(M, k=1)
        upper_vec = matrix[upper_idx]
        full_vec = matrix.reshape(-1)
        if zscore:
            upper_vec = _zscore(upper_vec)
            full_vec = _zscore(full_vec)
        vectors.append(
            SPIVectors(
                name=name,
                directed=bool(directed),
                upper=upper_vec.astype(np.float32, copy=False),
                full=full_vec.astype(np.float32, copy=False),
            )
        )
    return vectors


def _zscore(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    if vec.size == 0:
        return vec
    mean = float(np.mean(vec))
    std = float(np.std(vec))
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(vec)
    return (vec - mean) / std


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size:
        raise ValueError(
            f"Cannot correlate vectors with different sizes: {x.size} vs {y.size}"
        )
    if x.size < 2:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = spearmanr(x, y)
    corr = float(result.correlation if hasattr(result, "correlation") else result[0])
    if not np.isfinite(corr):
        return 0.0
    return corr


def compute_spi_similarity(vectors: Sequence[SPIVectors]) -> np.ndarray:
    n = len(vectors)
    corr = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = vectors[i].vector_pair(vectors[j])
            if vectors[i].directed != vectors[j].directed:
                # Fall back to the full flattened matrices so both vectors share dimensionality.
                vi, vj = vectors[i].full, vectors[j].full
            value = _safe_spearman(vi, vj)
            corr[i, j] = corr[j, i] = value
    np.fill_diagonal(corr, 1.0)
    return corr


def _flatten_upper(matrix: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(matrix, k=1)
    return matrix[iu].astype(np.float32, copy=False)


def process_mode(
    mode: str,
    *,
    recompute: bool,
    limit: int | None,
    zscore_vectors: bool,
) -> Path:
    dataset_records = list(enumerate_datasets(mode))
    if not dataset_records:
        raise RuntimeError(f"No datasets found under data/{mode}")
    if limit is not None:
        dataset_records = dataset_records[:limit]
    spi_names = dataset_records[0].spi_names
    spi_directed = dataset_records[0].spi_directed
    for record in dataset_records[1:]:
        if record.spi_names != spi_names:
            raise ValueError(
                f"Dataset {record.dataset_path} has a different SPI ordering."
            )
        if record.spi_directed != spi_directed:
            raise ValueError(
                f"Dataset {record.dataset_path} has different SPI metadata."
            )
    mode_results_dir = _mode_results_dir(mode)
    aggregated_features: list[np.ndarray] = []
    mts_classes: list[str] = []
    variants: list[str] = []
    Ms: list[int] = []
    Ts: list[int] = []
    dataset_paths: list[str] = []
    instances: list[int] = []
    for record in tqdm(dataset_records, desc=f"mode={mode}", unit="dataset"):
        arrays_dir = record.dataset_dir / "arrays"
        corr_path = arrays_dir / "spi_spi_corr.npy"
        if corr_path.exists() and not recompute:
            corr = np.load(corr_path)
        else:
            matrices = _load_spi_matrices(record)
            vectors = _vectorise_spi_matrices(record, matrices, zscore=zscore_vectors)
            corr = compute_spi_similarity(vectors)
            ensure_dir(arrays_dir)
            np.save(corr_path, corr.astype(np.float32))
        features = _flatten_upper(corr)
        feature_path = arrays_dir / "spi_spi_features.npy"
        np.save(feature_path, features)
        aggregated_features.append(features)
        mts_classes.append(record.mts_class)
        variants.append(record.variant)
        Ms.append(record.M)
        Ts.append(record.T)
        dataset_paths.append(record.dataset_path)
        instances.append(record.instance_index)
    X = np.vstack(aggregated_features)
    n_spis = len(spi_names)
    iu = np.triu_indices(n_spis, k=1)
    feature_pairs = np.column_stack(iu).astype(np.int16)
    result_path = mode_results_dir / f"spi_space_features_{mode}.npz"
    np.savez_compressed(
        result_path,
        X=X.astype(np.float32),
        mts_class=np.array(mts_classes, dtype=object),
        variant=np.array(variants, dtype=object),
        M=np.array(Ms, dtype=np.int32),
        T=np.array(Ts, dtype=np.int32),
        instance_index=np.array(instances, dtype=np.int32),
        dataset_paths=np.array(dataset_paths, dtype=object),
        spi_names=np.array(spi_names, dtype=object),
        spi_directed=np.array(spi_directed, dtype=bool),
        feature_pairs=feature_pairs,
    )
    LOGGER.info(
        "Saved %d x %d feature matrix to %s",
        X.shape[0],
        X.shape[1],
        result_path,
    )
    return result_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct SPI–SPI feature matrices for a dataset mode."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["dev", "full"],
        help="Dataset mode to process.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute SPI–SPI correlations even if cached.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    parser.add_argument(
        "--zscore",
        action="store_true",
        help="Apply per-SPI z-scoring to edge vectors before correlation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    process_mode(
        args.mode,
        recompute=args.recompute,
        limit=args.limit,
        zscore_vectors=args.zscore,
    )


if __name__ == "__main__":
    main()
