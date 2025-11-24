from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ConstantInputWarning, spearmanr
from tqdm import tqdm

from .utils import ensure_dir, iter_dataset_dirs, load_json, project_root, to_relative

LOGGER = logging.getLogger(__name__)

SPACE_CHOICES = ("spi-spi", "spi")


def _results_dir(mode: str, space: str, category: str) -> Path:
    if space not in SPACE_CHOICES:
        raise ValueError(f"Unsupported space '{space}'")
    base = ensure_dir(project_root() / "results" / mode / space)
    return ensure_dir(base / category)


def _format_param_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3g}".replace(".", "p").replace("-", "m")
    if isinstance(value, (int, np.integer)):
        return str(value)
    text = str(value).strip()
    return text.replace(" ", "")


def _build_display_class(meta: dict) -> str:
    base = meta.get("mts_class") or meta.get("class") or "unknown"
    variant = meta.get("variant")
    variant_name = ""
    params: dict[str, object] = {}
    if isinstance(variant, dict):
        variant_name = variant.get("name") or ""
        params = variant.get("params") or variant.get("values") or {}
    elif isinstance(variant, str):
        variant_name = variant
    if not params:
        generator = meta.get("generator") or {}
        params = generator.get("params") or {}
    param_parts: list[str] = []
    for key in sorted(params):
        value = _format_param_value(params[key])
        if value:
            param_parts.append(f"{key}{value}")
    suffix: list[str] = []
    if variant_name:
        suffix.append(variant_name)
    if param_parts:
        suffix.append("_".join(param_parts))
    if suffix:
        return f"{base}_{'_'.join(suffix)}"
    return base


def _plot_fingerprint(
    X: np.ndarray,
    dataset_paths: list[str],
    space: str,
    output_path: Path,
) -> None:
    if X.size == 0:
        return
    values = np.clip(X, -1.0, 1.0)
    height = max(1.5, X.shape[0] * 0.2)
    width = max(5.0, height * 5.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    sns.heatmap(
        values,
        cmap="coolwarm",
        ax=ax,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_title(f"{space} feature fingerprint")
    ax.set_xlabel("Features")
    ax.set_ylabel("Datasets")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _clean_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _drop_degenerate_pairs(
    X: np.ndarray,
    feature_pairs: np.ndarray,
    spi_names: list[str],
    features_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if X.size == 0:
        return X, feature_pairs
    span = np.ptp(X, axis=0)
    mask = span > 1e-8
    if mask.all():
        return X, feature_pairs
    dropped = feature_pairs[~mask]
    kept = feature_pairs[mask]
    X = X[:, mask]
    entries = [
        {
            "spi_i_index": int(i),
            "spi_j_index": int(j),
            "spi_i": spi_names[int(i)],
            "spi_j": spi_names[int(j)],
        }
        for i, j in dropped
    ]
    report_path = features_dir / "dropped_spi_pairs.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
    LOGGER.warning(
        "Dropped %d degenerate SPI–SPI columns (details in %s).",
        len(entries),
        report_path,
    )
    return X, kept


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
    display_class: str
    base_class: str


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
        display_class = _build_display_class(meta)
        base_class = meta.get("mts_class", meta.get("class", "unknown"))
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
            display_class=display_class,
            base_class=base_class,
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


def _spi_block_features(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    blocks: list[np.ndarray] = []
    for i in range(n):
        row = np.concatenate((matrix[i, : i], matrix[i, i + 1 :]))
        blocks.append(row)
    return np.concatenate(blocks).astype(np.float32, copy=False)


def _spi_feature_slices(n_spis: int) -> np.ndarray:
    length = n_spis - 1
    slices = np.zeros((n_spis, 2), dtype=np.int32)
    offset = 0
    for idx in range(n_spis):
        slices[idx] = (offset, length)
        offset += length
    return slices


def process_mode(
    mode: str,
    *,
    recompute: bool,
    limit: int | None,
    zscore_vectors: bool,
    space: str,
    fingerprint: bool,
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
    features_dir = _results_dir(mode, space, "features")
    aggregated_features: list[np.ndarray] = []
    mts_classes: list[str] = []
    variants: list[str] = []
    Ms: list[int] = []
    Ts: list[int] = []
    dataset_paths: list[str] = []
    instances: list[int] = []
    display_classes: list[str] = []
    base_classes: list[str] = []
    for record in tqdm(dataset_records, desc=f"mode={mode}-{space}", unit="dataset"):
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
        if space == "spi-spi":
            features = _flatten_upper(corr)
            feature_path = arrays_dir / "spi_spi_features.npy"
        elif space == "spi":
            features = _spi_block_features(corr)
            feature_path = arrays_dir / "spi_features.npy"
        else:
            raise ValueError(f"Unsupported space '{space}'")
        features = _clean_vector(features)
        np.save(feature_path, features)
        aggregated_features.append(features)
        mts_classes.append(record.mts_class)
        variants.append(record.variant)
        Ms.append(record.M)
        Ts.append(record.T)
        dataset_paths.append(record.dataset_path)
        instances.append(record.instance_index)
        display_classes.append(record.display_class)
        base_classes.append(record.base_class)
    X = np.vstack(aggregated_features)
    n_spis = len(spi_names)
    iu = np.triu_indices(n_spis, k=1)
    if space == "spi-spi":
        feature_pairs = np.column_stack(iu).astype(np.int16)
        spi_slices = np.empty((0, 2), dtype=np.int32)
    else:
        feature_pairs = np.empty((0, 2), dtype=np.int16)
        spi_slices = _spi_feature_slices(n_spis)
    if space == "spi-spi":
        X, feature_pairs = _drop_degenerate_pairs(
            X,
            feature_pairs,
            spi_names,
            features_dir,
        )
    result_path = features_dir / f"spi_space_features_{mode}.npz"
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
        spi_feature_slices=spi_slices,
        display_class=np.array(display_classes, dtype=object),
        base_class=np.array(base_classes, dtype=object),
        space=space,
    )
    LOGGER.info(
        "Saved %d x %d feature matrix to %s (%s space)",
        X.shape[0],
        X.shape[1],
        result_path,
        space,
    )
    if fingerprint:
        _plot_fingerprint(
            X,
            dataset_paths,
            space,
            features_dir / "spi_space_feature_matrix.png",
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
        "--space",
        choices=list(SPACE_CHOICES),
        default="spi-spi",
        help="Feature space to compute (default: spi-spi).",
    )
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help="Plot and save the dataset × feature fingerprint heatmap (spi-spi only).",
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
    if args.fingerprint and args.space != "spi-spi":
        raise ValueError("Fingerprint heatmap is only supported when --space=spi-spi.")
    process_mode(
        args.mode,
        recompute=args.recompute,
        limit=args.limit,
        zscore_vectors=args.zscore,
        space=args.space,
        fingerprint=args.fingerprint,
    )


if __name__ == "__main__":
    main()
