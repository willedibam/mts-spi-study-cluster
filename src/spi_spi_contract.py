"""Versioned SPI--SPI feature construction.

The primary unified contract compares complete ordered off-diagonal MPI
entries and emits exactly one value per unordered SPI pair.  The older
direction-expanded and legacy symmetrized contracts remain available for
sensitivity analyses and reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata


LEGACY_CONTRACT_VERSION = "legacy_symmetrized_v1"
DIRECTIONAL_CONTRACT_VERSION = "direction_preserving_v2"
UNIFIED_CONTRACT_VERSION = "unified_ordered_v3"
# Backwards-compatible alias for code that used CONTRACT_VERSION for v2.
CONTRACT_VERSION = DIRECTIONAL_CONTRACT_VERSION

Metric = Literal["pearson", "spearman", "mi"]


@dataclass(frozen=True)
class FeatureSpec:
    block: Literal["sym", "dir", "unified"]
    relation: Literal["sym", "parallel", "reverse", "reciprocity", "ordered"]
    spi_a: str
    spi_b: str

    def as_dict(self) -> dict[str, str]:
        return {
            "block": self.block,
            "relation": self.relation,
            "spi_a": self.spi_a,
            "spi_b": self.spi_b,
        }


@dataclass(frozen=True)
class FeatureBlocks:
    z_sym: np.ndarray
    z_dir: np.ndarray
    sym_schema: tuple[FeatureSpec, ...]
    dir_schema: tuple[FeatureSpec, ...]
    sym_valid: np.ndarray
    dir_valid: np.ndarray
    invalid_reasons: Mapping[str, Mapping[str, str]]

    @property
    def z_augmented(self) -> np.ndarray:
        return np.concatenate((self.z_sym, self.z_dir))

    @property
    def augmented_schema(self) -> tuple[FeatureSpec, ...]:
        return self.sym_schema + self.dir_schema


@dataclass(frozen=True)
class UnifiedFeatures:
    z: np.ndarray
    schema: tuple[FeatureSpec, ...]
    valid: np.ndarray
    invalid_reasons: Mapping[str, Mapping[str, str]]


def schema_json(schema: Sequence[FeatureSpec]) -> str:
    return json.dumps(
        [feature.as_dict() for feature in schema],
        sort_keys=True,
        separators=(",", ":"),
    )


def schema_sha256(schema: Sequence[FeatureSpec]) -> str:
    return hashlib.sha256(schema_json(schema).encode("utf-8")).hexdigest()


def _validate_mpi(name: str, matrix: np.ndarray, dimension: int | None) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} is not square (shape={matrix.shape})")
    if dimension is not None and matrix.shape != (dimension, dimension):
        raise ValueError(
            f"MPI dimension mismatch for {name}: expected {(dimension, dimension)}, "
            f"got {matrix.shape}"
        )
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} needs at least two channels")
    return matrix


def _upper_symmetrized(matrix: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(matrix.shape[0], k=1)
    return (0.5 * (matrix + matrix.T))[upper]


def _ordered_off_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Return ``A[i,j]`` for all ``i != j`` in C row-major order."""

    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def _vector_invalid_reason(vector: np.ndarray) -> str | None:
    if not np.isfinite(vector).all():
        return "nonfinite"
    centred = vector - vector.mean()
    if not np.isfinite(centred).all() or np.linalg.norm(centred) < 1e-12:
        return "constant"
    return None


def _pearson_corr_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    centred = vectors - vectors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    valid = (
        np.isfinite(centred).all(axis=1)
        & np.isfinite(norms)
        & (norms >= 1e-12)
    )
    normalized = np.zeros_like(centred, dtype=np.float64)
    normalized[valid] = centred[valid] / norms[valid, None]
    correlations = normalized @ normalized.T
    correlations[~valid, :] = np.nan
    correlations[:, ~valid] = np.nan
    return correlations


def _spearman_corr_matrix(vectors: np.ndarray) -> np.ndarray:
    ranks = rankdata(
        vectors,
        axis=1,
        method="average",
        nan_policy="propagate",
    )
    return _pearson_corr_matrix(ranks)


def _mutual_information_matrix(vectors: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Histogram normalized mutual information, with invalid rows as ``NaN``."""

    vectors = np.asarray(vectors, dtype=np.float64)
    n_vectors = vectors.shape[0]
    result = np.full((n_vectors, n_vectors), np.nan, dtype=np.float64)
    valid = np.array([_vector_invalid_reason(row) is None for row in vectors])
    digitized = np.zeros_like(vectors, dtype=np.int32)
    for index in np.flatnonzero(valid):
        row = vectors[index]
        bins = np.linspace(row.min(), row.max(), n_bins + 1)
        digitized[index] = np.clip(np.digitize(row, bins) - 1, 0, n_bins - 1)
    for first in np.flatnonzero(valid):
        result[first, first] = 1.0
        for second in np.flatnonzero(valid & (np.arange(n_vectors) > first)):
            joint = np.zeros((n_bins, n_bins), dtype=np.float64)
            np.add.at(joint, (digitized[first], digitized[second]), 1)
            joint /= joint.sum()
            marginal_first = joint.sum(axis=1)
            marginal_second = joint.sum(axis=0)
            independent = np.outer(marginal_first, marginal_second)
            occupied = (joint > 0) & (independent > 0)
            mutual_information = np.sum(
                joint[occupied] * np.log(joint[occupied] / independent[occupied])
            )
            entropy_first = -np.sum(
                marginal_first[marginal_first > 0]
                * np.log(marginal_first[marginal_first > 0])
            )
            entropy_second = -np.sum(
                marginal_second[marginal_second > 0]
                * np.log(marginal_second[marginal_second > 0])
            )
            scale = min(entropy_first, entropy_second)
            value = mutual_information / scale if scale > 1e-12 else np.nan
            result[first, second] = result[second, first] = value
    return result


def _similarity_matrix(vectors: np.ndarray, metric: Metric) -> np.ndarray:
    if metric == "pearson":
        return _pearson_corr_matrix(vectors)
    if metric == "spearman":
        return _spearman_corr_matrix(vectors)
    if metric == "mi":
        return _mutual_information_matrix(vectors)
    raise ValueError(f"unknown metric: {metric}")


def build_unified_features(
    mpis: Mapping[str, np.ndarray],
    spi_order: Sequence[str],
    *,
    metric: Metric = "pearson",
) -> UnifiedFeatures:
    """Return one aligned ordered-edge similarity per unordered SPI pair.

    Every MPI contributes ``A[i,j]`` for all ``i != j`` in row-major order.
    Comparing the same ordered positions preserves aligned direction for two
    directed SPIs, while treating directed and symmetric SPIs under one
    schema.  A common permutation of channel labels leaves all values
    unchanged.  Reverse-direction and self-reciprocity relations are not part
    of this contract.
    """

    z, valid, invalid_reasons = build_unified_feature_values(
        mpis,
        spi_order,
        metric=metric,
    )
    return UnifiedFeatures(
        z=z,
        schema=build_unified_schema(spi_order),
        valid=valid,
        invalid_reasons=invalid_reasons,
    )


def build_unified_schema(spi_order: Sequence[str]) -> tuple[FeatureSpec, ...]:
    """Construct the dataset-independent unified feature schema once."""

    if len(set(spi_order)) != len(spi_order):
        raise ValueError("spi_order contains duplicate names")
    return tuple(
        FeatureSpec("unified", "ordered", spi_order[first], spi_order[second])
        for first in range(len(spi_order))
        for second in range(first + 1, len(spi_order))
    )


def build_unified_feature_values(
    mpis: Mapping[str, np.ndarray],
    spi_order: Sequence[str],
    *,
    metric: Metric = "pearson",
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Mapping[str, str]]]:
    """Compute unified values without rebuilding the shared feature schema."""

    if len(set(spi_order)) != len(spi_order):
        raise ValueError("spi_order contains duplicate names")
    missing = [name for name in spi_order if name not in mpis]
    if missing:
        raise KeyError(f"missing MPI(s): {', '.join(missing)}")

    matrices: list[np.ndarray] = []
    dimension: int | None = None
    for name in spi_order:
        matrix = _validate_mpi(name, mpis[name], dimension)
        dimension = matrix.shape[0]
        matrices.append(matrix)

    ordered_vectors = np.vstack(
        [_ordered_off_diagonal(matrix) for matrix in matrices]
    )
    correlations = _similarity_matrix(ordered_vectors, metric)
    upper = np.triu_indices(len(spi_order), k=1)
    z = correlations[upper].astype(np.float32)
    invalid_reasons: dict[str, dict[str, str]] = {}
    for name, vector in zip(spi_order, ordered_vectors):
        reason = _vector_invalid_reason(vector)
        if reason is not None:
            invalid_reasons[name] = {"ordered": reason}
    return z, np.isfinite(z), invalid_reasons


def build_feature_blocks(
    mpis: Mapping[str, np.ndarray],
    spi_order: Sequence[str],
    directed_flags: Sequence[bool],
    *,
    metric: Metric = "pearson",
    include_reciprocity: bool = True,
) -> FeatureBlocks:
    """Construct invariant legacy and directional SPI--SPI feature blocks.

    ``z_sym`` is the legacy symmetrized block.  ``z_dir`` contains parallel
    ordered-edge correlations for every pair involving a directed SPI,
    reverse-edge correlations for directed--directed pairs, and (by default)
    directed-SPI self-reciprocity.  Undirected--undirected ordered features are
    exact duplicates of ``z_sym`` and are therefore omitted.
    """

    if len(spi_order) != len(directed_flags):
        raise ValueError("spi_order and directed_flags have different lengths")
    if len(set(spi_order)) != len(spi_order):
        raise ValueError("spi_order contains duplicate names")
    missing = [name for name in spi_order if name not in mpis]
    if missing:
        raise KeyError(f"missing MPI(s): {', '.join(missing)}")

    matrices: list[np.ndarray] = []
    dimension: int | None = None
    for name in spi_order:
        matrix = _validate_mpi(name, mpis[name], dimension)
        dimension = matrix.shape[0]
        matrices.append(matrix)

    sym_vectors = np.vstack([_upper_symmetrized(matrix) for matrix in matrices])
    ordered_vectors = np.vstack(
        [_ordered_off_diagonal(matrix) for matrix in matrices]
    )
    transposed_vectors = np.vstack(
        [_ordered_off_diagonal(matrix.T) for matrix in matrices]
    )

    sym_correlations = _similarity_matrix(sym_vectors, metric)
    ordered_correlations = _similarity_matrix(
        np.vstack((ordered_vectors, transposed_vectors)), metric
    )
    n_spis = len(spi_order)

    z_sym: list[float] = []
    sym_schema: list[FeatureSpec] = []
    z_dir: list[float] = []
    dir_schema: list[FeatureSpec] = []
    for first in range(n_spis):
        for second in range(first + 1, n_spis):
            name_first, name_second = spi_order[first], spi_order[second]
            z_sym.append(sym_correlations[first, second])
            sym_schema.append(
                FeatureSpec("sym", "sym", name_first, name_second)
            )
            if directed_flags[first] or directed_flags[second]:
                z_dir.append(ordered_correlations[first, second])
                dir_schema.append(
                    FeatureSpec("dir", "parallel", name_first, name_second)
                )
            if directed_flags[first] and directed_flags[second]:
                z_dir.append(ordered_correlations[first, n_spis + second])
                dir_schema.append(
                    FeatureSpec("dir", "reverse", name_first, name_second)
                )

    if include_reciprocity:
        for index, (name, directed) in enumerate(zip(spi_order, directed_flags)):
            if directed:
                z_dir.append(ordered_correlations[index, n_spis + index])
                dir_schema.append(FeatureSpec("dir", "reciprocity", name, name))

    invalid_reasons: dict[str, dict[str, str]] = {}
    for name, sym_vector, ordered_vector in zip(
        spi_order, sym_vectors, ordered_vectors
    ):
        reasons: dict[str, str] = {}
        sym_reason = _vector_invalid_reason(sym_vector)
        ordered_reason = _vector_invalid_reason(ordered_vector)
        if sym_reason is not None:
            reasons["sym"] = sym_reason
        if ordered_reason is not None:
            reasons["ordered"] = ordered_reason
        if reasons:
            invalid_reasons[name] = reasons

    sym_array = np.asarray(z_sym, dtype=np.float32)
    dir_array = np.asarray(z_dir, dtype=np.float32)
    return FeatureBlocks(
        z_sym=sym_array,
        z_dir=dir_array,
        sym_schema=tuple(sym_schema),
        dir_schema=tuple(dir_schema),
        sym_valid=np.isfinite(sym_array),
        dir_valid=np.isfinite(dir_array),
        invalid_reasons=invalid_reasons,
    )
