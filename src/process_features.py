"""
Compute and cache SPI-SPI feature matrices for downstream analysis.

- ``legacy_symmetrized_v1`` exactly preserves the historical construction.
- ``unified_ordered_v3`` (default) emits one ordered-entry correlation per
  unordered SPI pair: exactly ``K choose 2`` features.
- ``direction_preserving_v2`` emits a frozen ``z_sym`` block plus a
  channel-permutation-invariant directional block over ordered off-diagonals.
- Canonical v2 artifacts retain undefined correlations as NaN and never apply
  corpus-wide variance filtering.
- Supports optional SPI name subset via --spi-subset (txt, one per line).
- Supports different metrics via --metric (comma-separated): spearman, pearson, mi (mutual information).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import logging
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, ConstantInputWarning

from .utils import load_json, project_root
from .spi_spi_contract import (
    DIRECTIONAL_CONTRACT_VERSION,
    LEGACY_CONTRACT_VERSION,
    UNIFIED_CONTRACT_VERSION,
    FeatureSpec,
    build_feature_blocks,
    build_unified_feature_values,
    build_unified_schema,
    schema_sha256,
)

import warnings

warnings.simplefilter("ignore", ConstantInputWarning)
LOGGER = logging.getLogger(__name__)

MetricType = Literal["spearman", "pearson", "mi"]
NonFinitePolicy = Literal["zero", "nan", "raise"]
FeatureContract = Literal[
    "legacy_symmetrized_v1",
    "direction_preserving_v2",
    "unified_ordered_v3",
]


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
            samples.append(
                {
                    "label": meta["mts_class"],
                    "labels": meta.get("labels", []),
                    "M": meta.get("M"),
                    "T": meta.get("T"),
                    "path": ds_dir,
                    "variant": (meta.get("variant") or {}).get("name", "") if isinstance(meta.get("variant"), dict) else (meta.get("variant") or ""),
                    "instance": meta.get("instance_index"),
                    "meta_path": meta_path,
                    "pyspi_provenance": {
                        "config": meta.get("pyspi", {}).get("config"),
                        "config_sha256": meta.get("pyspi", {}).get("config_sha256"),
                        "version": meta.get("pyspi", {}).get("version"),
                        "normalise": meta.get("normalise"),
                    },
                    "experiment_provenance": meta.get("experiment", {}),
                }
            )
            if limit and len(samples) >= limit:
                break
        if limit and len(samples) >= limit:
            break
    if spi_order is None or directed_flags is None:
        raise RuntimeError(f"No datasets found for data_path={data_path}")
    return samples, spi_order, directed_flags


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _repository_state() -> dict[str, object]:
    root = project_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {"git_commit": commit, "git_dirty": dirty}


def validate_source_provenance(samples: Sequence[Dict]) -> dict[str, object]:
    """Validate computation-level provenance before datasets are pooled."""

    if not samples:
        raise ValueError("samples is empty")
    fields = ("config_sha256", "config", "version", "normalise")
    signatures: dict[str, set[str]] = {field: set() for field in fields}
    incomplete: set[str] = set()
    for sample in samples:
        provenance = sample.get("pyspi_provenance", {})
        for field in fields:
            value = provenance.get(field)
            if value is None:
                incomplete.add(field)
            else:
                signatures[field].add(_canonical_json(value))

    # A hash is authoritative when available; a config path is only a fallback
    # identity for older archives that did not record content hashes.
    authoritative_config = (
        signatures["config_sha256"]
        if signatures["config_sha256"]
        else signatures["config"]
    )
    if len(authoritative_config) > 1:
        raise ValueError("pooled datasets have different pyspi configurations")
    if len(signatures["version"]) > 1:
        raise ValueError("pooled datasets have different pyspi computation versions")
    if len(signatures["normalise"]) > 1:
        raise ValueError("pooled datasets have different pyspi normalization settings")

    return {
        "config_sha256": sorted(signatures["config_sha256"]),
        "config": sorted(signatures["config"]),
        "version": sorted(signatures["version"]),
        "normalise": sorted(signatures["normalise"]),
        "status": "incomplete" if incomplete else "complete",
        "missing_fields": sorted(incomplete),
    }


def build_source_manifest(samples: Sequence[Dict]) -> dict[str, object]:
    """Build a content-addressed identity for metadata and MPI inputs."""

    entries: list[dict[str, object]] = []
    for sample in samples:
        dataset_path = Path(sample["path"])
        meta_path = Path(sample.get("meta_path", dataset_path / "meta.json"))
        mpi_path = dataset_path / "spi_mpis.npz"
        if not mpi_path.exists():
            raise FileNotFoundError(f"MPI archive not found: {mpi_path}")
        entries.append(
            {
                "dataset_path": str(dataset_path.resolve()),
                "meta_sha256": _file_sha256(meta_path),
                "mpi_sha256": _file_sha256(mpi_path),
                "experiment": sample.get("experiment_provenance", {}),
            }
        )
    manifest_json = _canonical_json(entries)
    return {
        "entries": entries,
        "sha256": hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
    }


def _edge_vectors(
    name: str,
    mat: np.ndarray,
    directed: bool,
    split_directed: bool = False,
) -> List[tuple[str, np.ndarray]]:
    mat = np.asarray(mat, float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"{name} is not square (shape={mat.shape})")
    if not directed or not split_directed:
        mat = 0.5 * (mat + mat.T)
        mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
        return [(name, mat[mask])]
    upper_mask = np.triu(np.ones(mat.shape, dtype=bool), k=1)
    return [
        (f"{name}__ij", mat[upper_mask]),
        # Transpose before applying the same mask so both vectors enumerate
        # the same unordered dyads: (0,1), (0,2), ..., (M-2,M-1).
        (f"{name}__ji", mat.T[upper_mask]),
    ]


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Average-tie ranks along rows; invalid rows remain invalid."""
    return rankdata(arr, axis=1, method="average", nan_policy="propagate")


def _pearson_corr_matrix(V: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix for row vectors."""
    V = V - V.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(V, axis=1)
    valid = np.isfinite(V).all(axis=1) & np.isfinite(norms) & (norms >= 1e-12)
    normalized = np.zeros_like(V, dtype=np.float64)
    normalized[valid] = V[valid] / norms[valid, None]
    corr = normalized @ normalized.T
    corr[~valid, :] = np.nan
    corr[:, ~valid] = np.nan
    return corr


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
    metric: MetricType = "pearson",
    nonfinite_policy: NonFinitePolicy = "zero",
) -> tuple[np.ndarray, List[str]]:
    with np.load(sample["path"] / "spi_mpis.npz") as npz:
        mpis = {k: npz[k] for k in npz.files}
    vectors: List[np.ndarray] = []
    names: List[str] = []
    for name, directed in zip(spi_order, directed_flags):
        entries = _edge_vectors(name, mpis[name], directed, split_directed)
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
    
    if nonfinite_policy == "raise" and not np.isfinite(corr).all():
        invalid = int((~np.isfinite(corr)).sum())
        raise ValueError(f"SPI-SPI matrix contains {invalid} non-finite values")
    if nonfinite_policy == "zero":
        corr = np.where(np.isfinite(corr), corr, 0.0)
    elif nonfinite_policy != "nan":
        raise ValueError(
            f"unknown nonfinite_policy {nonfinite_policy!r}; expected 'zero', 'nan' or 'raise'"
        )
    corr = corr.astype(np.float32)
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


def build_direction_preserving_feature_matrix(
    samples: List[Dict],
    spi_order: List[str],
    directed_flags: List[bool],
    *,
    metric: MetricType = "pearson",
    include_reciprocity: bool = True,
    workers: int = 1,
) -> dict[str, object]:
    """Build the complete v2 schema without corpus-dependent filtering."""

    sym_rows: list[np.ndarray] = []
    dir_rows: list[np.ndarray] = []
    sym_valid_rows: list[np.ndarray] = []
    dir_valid_rows: list[np.ndarray] = []
    invalid_reasons: list[str] = []
    sym_schema_ref: tuple[FeatureSpec, ...] | None = None
    dir_schema_ref: tuple[FeatureSpec, ...] | None = None
    tasks = [
        (
            str(sample["path"]),
            tuple(spi_order),
            tuple(directed_flags),
            metric,
            include_reciprocity,
        )
        for sample in samples
    ]
    if workers < 1:
        raise ValueError("workers must be at least one")
    if workers == 1:
        results = map(_build_direction_preserving_sample, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
        )
        results = executor.map(_build_direction_preserving_sample, tasks)
    try:
        for index, result in enumerate(results, start=1):
            if sym_schema_ref is None:
                sym_schema_ref = result.sym_schema
                dir_schema_ref = result.dir_schema
            elif (
                result.sym_schema != sym_schema_ref
                or result.dir_schema != dir_schema_ref
            ):
                raise ValueError("feature schema mismatch across datasets")
            sym_rows.append(result.z_sym)
            dir_rows.append(result.z_dir)
            sym_valid_rows.append(result.sym_valid)
            dir_valid_rows.append(result.dir_valid)
            invalid_reasons.append(_canonical_json(result.invalid_reasons))
            if index % 10 == 0 or index == len(samples):
                LOGGER.info(
                    "direction-preserving features: %d/%d (%.0f%%)",
                    index,
                    len(samples),
                    100 * index / len(samples),
                )
    finally:
        if executor is not None:
            executor.shutdown()
    if sym_schema_ref is None or dir_schema_ref is None:
        raise RuntimeError("No features computed")

    z_sym = np.vstack(sym_rows)
    z_dir = np.vstack(dir_rows) if dir_rows[0].size else np.empty((len(samples), 0), dtype=np.float32)
    sym_valid = np.vstack(sym_valid_rows)
    dir_valid = (
        np.vstack(dir_valid_rows)
        if dir_valid_rows[0].size
        else np.empty((len(samples), 0), dtype=bool)
    )
    schema = sym_schema_ref + dir_schema_ref
    return {
        "X_sym": z_sym,
        "X_dir": z_dir,
        "sym_validity_mask": sym_valid,
        "dir_validity_mask": dir_valid,
        "invalid_reasons_json": np.asarray(invalid_reasons, dtype=object),
        "schema": schema,
        "sym_schema": sym_schema_ref,
        "dir_schema": dir_schema_ref,
    }


def build_unified_feature_matrix(
    samples: List[Dict],
    spi_order: List[str],
    *,
    metric: MetricType = "pearson",
    workers: int = 1,
) -> dict[str, object]:
    """Build the unified, complete ``K choose 2`` feature matrix."""

    rows: list[np.ndarray] = []
    valid_rows: list[np.ndarray] = []
    invalid_reasons: list[str] = []
    schema = build_unified_schema(spi_order)
    tasks = [
        (str(sample["path"]), tuple(spi_order), metric)
        for sample in samples
    ]
    if workers < 1:
        raise ValueError("workers must be at least one")
    if workers == 1:
        results = map(_build_unified_sample, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
        )
        results = executor.map(_build_unified_sample, tasks)
    try:
        for index, (values, valid, reasons) in enumerate(results, start=1):
            rows.append(values)
            valid_rows.append(valid)
            invalid_reasons.append(_canonical_json(reasons))
            if index % 10 == 0 or index == len(samples):
                LOGGER.info(
                    "unified features: %d/%d (%.0f%%)",
                    index,
                    len(samples),
                    100 * index / len(samples),
                )
    finally:
        if executor is not None:
            executor.shutdown()
    if not rows:
        raise RuntimeError("No features computed")
    return {
        "X": np.vstack(rows),
        "validity_mask": np.vstack(valid_rows),
        "invalid_reasons_json": np.asarray(invalid_reasons),
        "schema": schema,
    }


def _build_direction_preserving_sample(
    task: tuple[str, tuple[str, ...], tuple[bool, ...], MetricType, bool],
):
    dataset_path, spi_order, directed_flags, metric, include_reciprocity = task
    with np.load(Path(dataset_path) / "spi_mpis.npz") as archive:
        mpis = {name: archive[name] for name in spi_order}
    return build_feature_blocks(
        mpis,
        spi_order,
        directed_flags,
        metric=metric,
        include_reciprocity=include_reciprocity,
    )


def _build_unified_sample(
    task: tuple[str, tuple[str, ...], MetricType],
):
    dataset_path, spi_order, metric = task
    with np.load(Path(dataset_path) / "spi_mpis.npz") as archive:
        mpis = {name: archive[name] for name in spi_order}
    return build_unified_feature_values(mpis, spi_order, metric=metric)


def cache_path(
    data_path: str,
    limit: int | None,
    subset_label: str | None,
    *,
    split_directed: bool = False,
    metric: MetricType = "spearman",
    output_dir: str | None = None,
    feature_contract: FeatureContract = UNIFIED_CONTRACT_VERSION,
) -> Path:
    suffix = f"_limit{limit}" if limit else ""
    subset_suffix = f"_{subset_label}" if subset_label else ""
    split_suffix = "_split" if split_directed else ""
    metric_suffix = f"_{metric}"
    contract_suffix = {
        LEGACY_CONTRACT_VERSION: "",
        DIRECTIONAL_CONTRACT_VERSION: "_direction-v2",
        UNIFIED_CONTRACT_VERSION: "_unified-v3",
    }[feature_contract]
    safe = data_path.replace("\\", "-").replace("/", "-").strip("-")
    base_dir = Path(output_dir) if output_dir else project_root() / "features"
    return base_dir / f"{safe}{contract_suffix}{split_suffix}{metric_suffix}{subset_suffix}{suffix}.npz"


def load_cached_features(
    path: Path,
    recompute: bool,
    *,
    expected_cache_identity: str | None = None,
) -> dict | None:
    if recompute or not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        payload = {k: data[k] for k in data.files}
    if expected_cache_identity is not None:
        actual = payload.get("cache_identity_json")
        if actual is None:
            raise ValueError(
                f"cache {path} predates validated cache identity; use --recompute"
            )
        actual_text = str(np.asarray(actual).item())
        if actual_text != expected_cache_identity:
            raise ValueError(
                f"cache identity mismatch for {path}; use --recompute or a new output"
            )
    contract_value = payload.get("feature_contract")
    contract = str(np.asarray(contract_value).item()) if contract_value is not None else None
    if contract == DIRECTIONAL_CONTRACT_VERSION:
        required = (
            "X_sym",
            "X_dir",
            "sym_validity_mask",
            "dir_validity_mask",
            "feature_block",
            "feature_relation",
            "feature_spi_a",
            "feature_spi_b",
            "schema_sha256",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"cache {path} missing required fields: {', '.join(missing)}")
        schema = tuple(
            FeatureSpec(str(block), str(relation), str(first), str(second))
            for block, relation, first, second in zip(
                payload["feature_block"],
                payload["feature_relation"],
                payload["feature_spi_a"],
                payload["feature_spi_b"],
            )
        )
        expected_schema_hash = str(np.asarray(payload["schema_sha256"]).item())
        if schema_sha256(schema) != expected_schema_hash:
            raise ValueError(f"cache {path} has a corrupt feature schema hash")
        feature_count = payload["X_sym"].shape[1] + payload["X_dir"].shape[1]
        if feature_count != len(schema):
            raise ValueError(f"cache {path} feature matrix/schema dimensions disagree")
        if payload["sym_validity_mask"].shape != payload["X_sym"].shape:
            raise ValueError(f"cache {path} symmetric validity mask dimensions disagree")
        if payload["dir_validity_mask"].shape != payload["X_dir"].shape:
            raise ValueError(f"cache {path} directional validity mask dimensions disagree")
        if not np.array_equal(
            payload["sym_validity_mask"], np.isfinite(payload["X_sym"])
        ) or not np.array_equal(
            payload["dir_validity_mask"], np.isfinite(payload["X_dir"])
        ):
            raise ValueError(f"cache {path} validity masks do not match stored values")
    elif contract == UNIFIED_CONTRACT_VERSION:
        required = (
            "X",
            "validity_mask",
            "feature_block",
            "feature_relation",
            "feature_spi_a",
            "feature_spi_b",
            "schema_sha256",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"cache {path} missing required fields: {', '.join(missing)}")
        schema = tuple(
            FeatureSpec(str(block), str(relation), str(first), str(second))
            for block, relation, first, second in zip(
                payload["feature_block"],
                payload["feature_relation"],
                payload["feature_spi_a"],
                payload["feature_spi_b"],
            )
        )
        expected_schema_hash = str(np.asarray(payload["schema_sha256"]).item())
        if schema_sha256(schema) != expected_schema_hash:
            raise ValueError(f"cache {path} has a corrupt feature schema hash")
        if payload["X"].shape[1] != len(schema):
            raise ValueError(f"cache {path} feature matrix/schema dimensions disagree")
        if payload["validity_mask"].shape != payload["X"].shape:
            raise ValueError(f"cache {path} validity mask dimensions disagree")
        if not np.array_equal(payload["validity_mask"], np.isfinite(payload["X"])):
            raise ValueError(f"cache {path} validity mask does not match stored values")
    return payload


def save_cached_features(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
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
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for cached features (default: features/).",
    )
    parser.add_argument("--spi-subset", type=str, default=None, help="Path to txt file listing SPI names (one per line).")
    parser.add_argument("--recompute", action="store_true", help="Recompute even if cache exists.")
    parser.add_argument(
        "--feature-contract",
        choices=(
            LEGACY_CONTRACT_VERSION,
            DIRECTIONAL_CONTRACT_VERSION,
            UNIFIED_CONTRACT_VERSION,
        ),
        default=UNIFIED_CONTRACT_VERSION,
        help=(
            "Feature contract. Default: unified_ordered_v3, with exactly one "
            "ordered-entry similarity per unordered SPI pair."
        ),
    )
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
        default="pearson",
        help="Comma-separated metrics to compute: pearson (default), spearman, mi.",
    )
    parser.add_argument(
        "--var-threshold",
        type=float,
        default=None,
        help=(
            "Legacy-only corpus variance filter. Omit for unified/v2; fit "
            "filtering downstream. Historical default was 1e-8."
        ),
    )
    parser.add_argument(
        "--no-reciprocity",
        action="store_true",
        help="Omit directed-SPI self-reciprocity from the v2 sensitivity block.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Dataset-level worker processes for unified/v2 reconstruction (default: 1).",
    )
    return parser.parse_args(argv)


def _parse_metrics(raw: str | None) -> List[MetricType]:
    if not raw:
        return ["pearson"]
    metrics = [m.strip().lower() for m in raw.split(",") if m.strip()]
    if not metrics:
        return ["pearson"]
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
    
    contract: FeatureContract = args.feature_contract
    if contract != LEGACY_CONTRACT_VERSION and args.split_directed:
        raise ValueError(
            "--split-directed is a legacy fixed-label construction and is not "
            "valid under unified/v2 contracts"
        )
    if contract != LEGACY_CONTRACT_VERSION and args.var_threshold is not None:
        raise ValueError(
            "unified/v2 contracts never apply a corpus variance filter; "
            "fit filtering downstream on development rows"
        )
    if contract != DIRECTIONAL_CONTRACT_VERSION and args.no_reciprocity:
        raise ValueError("--no-reciprocity applies only to direction_preserving_v2")
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    if contract == LEGACY_CONTRACT_VERSION and args.workers != 1:
        raise ValueError("--workers currently applies only to unified/v2 contracts")

    metrics = _parse_metrics(args.metric)
    multi_metric = len(metrics) > 1
    LOGGER.info("Using metric(s): %s", ", ".join(metrics))

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
    pyspi_provenance = validate_source_provenance(samples)
    source_manifest = build_source_manifest(samples)
    repository = _repository_state()
    builder_identity = {
        "process_features_sha256": _file_sha256(Path(__file__)),
        "spi_spi_contract_sha256": _file_sha256(
            Path(__file__).with_name("spi_spi_contract.py")
        ),
        **repository,
    }

    targets: List[tuple[MetricType, Path, str]] = []
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
                output_dir=args.output_dir,
                feature_contract=contract,
            )
        )
        cache_identity = _canonical_json(
            {
                "feature_contract": contract,
                "metric": metric,
                "nonfinite_policy": (
                    "zero" if contract == LEGACY_CONTRACT_VERSION else "nan"
                ),
                "split_directed": bool(args.split_directed),
                "include_reciprocity": (
                    not args.no_reciprocity
                    if contract == DIRECTIONAL_CONTRACT_VERSION
                    else None
                ),
                "legacy_var_threshold": (
                    args.var_threshold
                    if args.var_threshold is not None
                    else 1e-8
                    if contract == LEGACY_CONTRACT_VERSION
                    else None
                ),
                "spi_order_sha256": hashlib.sha256(
                    _canonical_json(spi_order).encode("utf-8")
                ).hexdigest(),
                "directed_flags_sha256": hashlib.sha256(
                    _canonical_json(directed_flags).encode("utf-8")
                ).hexdigest(),
                "source_manifest_sha256": source_manifest["sha256"],
                "pyspi": pyspi_provenance,
                "builder": builder_identity,
            }
        )
        cached = load_cached_features(
            cache_file,
            recompute=args.recompute,
            expected_cache_identity=cache_identity,
        )
        if cached:
            LOGGER.info("Validated cache exists, skipping computation: %s", cache_file)
        else:
            targets.append((metric, cache_file, cache_identity))

    sample_payload = {
        "y": np.asarray([sample["label"] for sample in samples]),
        "labels": np.asarray([sample.get("labels", []) for sample in samples], dtype=object),
        "dataset_paths": np.asarray([str(sample["path"]) for sample in samples], dtype=object),
        "variant": np.asarray([sample.get("variant", "") for sample in samples], dtype=object),
        "M": np.asarray([sample.get("M") for sample in samples], dtype=object),
        "T": np.asarray([sample.get("T") for sample in samples], dtype=object),
        "instance": np.asarray([sample.get("instance") for sample in samples], dtype=object),
    }
    for metric, cache_file, cache_identity in targets:
        LOGGER.info("Computing %s features (metric=%s)", contract, metric)
        common_payload = {
            **sample_payload,
            "spi_order": np.asarray(spi_order, dtype=object),
            "directed_flags": np.asarray(directed_flags, dtype=bool),
            "mode": args.data_path,
            "dataset_limit": args.dataset_limit if args.dataset_limit is not None else -1,
            "spi_subset": subset_label or "",
            "metric": metric,
            "feature_contract": contract,
            "cache_identity_json": cache_identity,
            "source_manifest_json": _canonical_json(source_manifest),
            "pyspi_provenance_json": _canonical_json(pyspi_provenance),
            "builder_provenance_json": _canonical_json(builder_identity),
        }
        if contract == LEGACY_CONTRACT_VERSION:
            X_raw, _, pairs, _, _, _, _, _ = build_feature_matrix(
                samples,
                spi_order,
                directed_flags,
                split_directed=args.split_directed,
                metric=metric,
            )
            pairs_arr = np.asarray(pairs, dtype=object)
            threshold = args.var_threshold if args.var_threshold is not None else 1e-8
            if threshold > 0:
                keep = np.std(X_raw, axis=0) >= threshold
                X_raw = X_raw[:, keep]
                pairs_arr = pairs_arr[keep]
            payload = {
                **common_payload,
                "X": X_raw.astype(np.float32),
                "pairs": pairs_arr,
                "split_directed": bool(args.split_directed),
                "legacy_var_threshold": threshold,
            }
        elif contract == DIRECTIONAL_CONTRACT_VERSION:
            result = build_direction_preserving_feature_matrix(
                samples,
                spi_order,
                directed_flags,
                metric=metric,
                include_reciprocity=not args.no_reciprocity,
                workers=args.workers,
            )
            schema = result.pop("schema")
            sym_schema = result.pop("sym_schema")
            dir_schema = result.pop("dir_schema")
            payload = {
                **common_payload,
                **result,
                "feature_block": np.asarray(
                    [feature.block for feature in schema], dtype=object
                ),
                "feature_relation": np.asarray(
                    [feature.relation for feature in schema], dtype=object
                ),
                "feature_spi_a": np.asarray(
                    [feature.spi_a for feature in schema], dtype=object
                ),
                "feature_spi_b": np.asarray(
                    [feature.spi_b for feature in schema], dtype=object
                ),
                "schema_sha256": schema_sha256(schema),
                "sym_schema_sha256": schema_sha256(sym_schema),
                "dir_schema_sha256": schema_sha256(dir_schema),
                "nonfinite_policy": "nan",
                "diagonal_policy": "excluded",
                "ordered_entry_order": "C-row-major-i-ne-j",
                "include_reciprocity": not args.no_reciprocity,
            }
        else:
            result = build_unified_feature_matrix(
                samples,
                spi_order,
                metric=metric,
                workers=args.workers,
            )
            schema = result.pop("schema")
            payload = {
                **common_payload,
                **result,
                "feature_block": np.asarray(
                    [feature.block for feature in schema], dtype=object
                ),
                "feature_relation": np.asarray(
                    [feature.relation for feature in schema], dtype=object
                ),
                "feature_spi_a": np.asarray(
                    [feature.spi_a for feature in schema], dtype=object
                ),
                "feature_spi_b": np.asarray(
                    [feature.spi_b for feature in schema], dtype=object
                ),
                "schema_sha256": schema_sha256(schema),
                "nonfinite_policy": "nan",
                "diagonal_policy": "excluded",
                "ordered_entry_order": "C-row-major-i-ne-j",
            }
        save_cached_features(cache_file, payload)


if __name__ == "__main__":
    main()
