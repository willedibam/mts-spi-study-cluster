"""Catalogue-matched MPI summaries; no corpus fitting or NaN imputation.

Rows and columns retain the archive convention, without assuming a causal
source/target interpretation. All summaries exclude MPI diagonal entries.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


MARGINAL_NAMES = ("mean", "std", "q10", "q25", "median", "q75", "q90")
GRAPH_NAMES = (
    "row_mean_std", "column_mean_std", "row_mean_q10", "row_mean_q90",
    "column_mean_q10", "column_mean_q90", "reciprocity",
    "largest_singular_energy_fraction", "normalized_singular_effective_rank",
)


def summarize_mpis(
    mpis: Mapping[str, np.ndarray], spi_order: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened marginal/graph blocks and SPI correlation-valid mask.

    Any nonfinite off-diagonal entry invalidates that SPI's entire summary.
    Finite constant SPIs have meaningful marginal/strength summaries, but their
    Pearson correlations are undefined. Zero matrices also have undefined
    normalized singular-energy summaries. These distinctions remain NaN.
    """
    if not spi_order or len(set(spi_order)) != len(spi_order):
        raise ValueError("spi_order must be nonempty and unique")
    marginal = np.full((len(spi_order), len(MARGINAL_NAMES)), np.nan)
    graph = np.full((len(spi_order), len(GRAPH_NAMES)), np.nan)
    correlation_valid = np.zeros(len(spi_order), dtype=bool)
    dimension = None
    for index, name in enumerate(spi_order):
        matrix = np.asarray(mpis[name], dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 2:
            raise ValueError(f"{name}: expected a square MPI with M >= 2")
        if dimension is not None and len(matrix) != dimension:
            raise ValueError("MPI dimensions differ")
        dimension = len(matrix)
        mask = ~np.eye(dimension, dtype=bool)
        edges = matrix[mask]
        if not np.isfinite(edges).all():
            continue
        centred = edges - edges.mean()
        norm = np.linalg.norm(centred)
        correlation_valid[index] = np.isfinite(norm) and norm >= 1e-12
        marginal[index] = [edges.mean(), edges.std(ddof=0),
                           *np.quantile(edges, [0.1, 0.25, 0.5, 0.75, 0.9])]
        offdiag = np.where(mask, matrix, 0.0)
        row = offdiag.sum(axis=1) / (dimension - 1)
        column = offdiag.sum(axis=0) / (dimension - 1)
        graph[index, :6] = [row.std(), column.std(), *np.quantile(row, [0.1, 0.9]),
                            *np.quantile(column, [0.1, 0.9])]
        if correlation_valid[index]:
            reverse = offdiag.T[mask] - edges.mean()
            graph[index, 6] = (centred / norm) @ (reverse / norm)
        # Rescaling before SVD avoids overflow without changing energy fractions.
        maximum = np.max(np.abs(edges))
        if maximum > 0:
            singular = np.linalg.svd(offdiag / maximum, compute_uv=False)
            energy = singular**2 / np.sum(singular**2)
            positive = energy[energy > 0]
            graph[index, 7:] = [energy[0],
                np.exp(-np.sum(positive * np.log(positive))) / dimension]
    return marginal.ravel(), graph.ravel(), correlation_valid


def pearson_geometry_audit(mpis: Mapping[str, np.ndarray], spi_order: Sequence[str]) -> dict:
    """Audit only each record's valid principal block, without filling entries.

    Blocks can have different SPI identities: their spectra are diagnostics,
    not directly comparable fixed-coordinate representations.
    """
    first = np.asarray(mpis[spi_order[0]])
    mask = ~np.eye(len(first), dtype=bool)
    vectors = np.asarray([np.asarray(mpis[name])[mask] for name in spi_order])
    finite = np.isfinite(vectors).all(axis=1)
    vectors = vectors[finite]
    centred = vectors - vectors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    valid = np.isfinite(norms) & (norms >= 1e-12)
    normalized = centred[valid] / norms[valid, None]
    if not len(normalized):
        return {"valid_spis": 0, "rank_upper_bound": 0, "numerical_rank": 0,
                "minimum_eigenvalue": None, "effective_rank": None}
    eigen = np.linalg.eigvalsh(normalized @ normalized.T)
    positive = np.maximum(eigen, 0)
    probability = positive[positive > 0] / positive.sum()
    return {
        "valid_spis": len(normalized),
        "rank_upper_bound": int(min(len(normalized), mask.sum() - 1)),
        "numerical_rank": int(np.sum(eigen > 1e-8 * eigen[-1])),
        "rank_relative_tolerance": 1e-8,
        "minimum_eigenvalue": float(eigen[0]),
        "effective_rank": float(np.exp(-np.sum(probability * np.log(probability)))),
    }
