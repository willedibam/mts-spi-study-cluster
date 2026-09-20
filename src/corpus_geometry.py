"""Reusable geometry for SPI--SPI dataset representations.

The module keeps the historical recipe available for reproduction, but makes
the fitted preprocessing state explicit for new analyses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


Scaling = Literal["center", "standard", "robust"]


@dataclass(frozen=True)
class FrozenGeometryTransform:
    """Feature selection, imputation and scaling fitted on reference rows."""

    keep_indices: np.ndarray
    valid_fraction: np.ndarray
    impute_values: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    scaling: Scaling
    minimum_valid_fraction: float
    variance_threshold: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        selected = np.asarray(values, dtype=np.float64)[:, self.keep_indices]
        filled = np.where(np.isfinite(selected), selected, self.impute_values)
        return (filled - self.location) / self.scale


@dataclass(frozen=True)
class LegacyGeometry:
    """Output of the old transductive z-score/filter/zero-fill recipe."""

    values: np.ndarray
    row_mask: np.ndarray
    feature_mask: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    minimum_feature_valid_fraction: float
    minimum_row_valid_fraction: float


def load_feature_rows(
    paths: Sequence[str | Path], matrix_key: str = "auto"
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load and concatenate compatible feature artifacts.

    ``matrix_key='auto'`` selects ``X`` for unified-v3 artifacts and ``X_sym``
    for historical direction-preserving-v2 proof artifacts.
    """

    if not paths:
        raise ValueError("at least one feature artifact is required")
    matrices: list[np.ndarray] = []
    fields: dict[str, list[np.ndarray]] = {
        name: [] for name in ("y", "labels", "dataset_paths", "M", "T", "instance")
    }
    schema: tuple[str, tuple[int, ...]] | None = None
    for raw_path in paths:
        path = Path(raw_path)
        with np.load(path, allow_pickle=True) as archive:
            key = matrix_key
            if key == "auto":
                key = "X" if "X" in archive.files else "X_sym"
            if key not in archive.files:
                raise KeyError(f"{path} does not contain {key!r}")
            matrix = np.asarray(archive[key], dtype=np.float64)
            if matrix.ndim != 2:
                raise ValueError(f"{path}:{key} is not two-dimensional")
            identity = (key, (matrix.shape[1],))
            if schema is not None and identity != schema:
                raise ValueError("feature artifacts have incompatible matrix schemas")
            schema = identity
            matrices.append(matrix)
            for name in fields:
                if name in archive.files:
                    fields[name].append(np.asarray(archive[name]))
                elif name == "y":
                    fields[name].append(np.asarray([f"row-{i}" for i in range(len(matrix))]))
                else:
                    fields[name].append(np.full(len(matrix), None, dtype=object))
    metadata = {name: np.concatenate(parts) for name, parts in fields.items()}
    return np.concatenate(matrices, axis=0), metadata


def legacy_geometry(
    values: np.ndarray,
    *,
    minimum_feature_valid_fraction: float = 0.90,
    minimum_row_valid_fraction: float = 0.80,
) -> LegacyGeometry:
    """Mirror ``old/compute_distance_matrix.py`` on a rows-by-features array.

    Statistics are deliberately fitted on every supplied row. Remaining NaNs
    are replaced by zero after z-scoring, which is mean imputation in the
    standardized coordinate system.
    """

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must be two-dimensional")
    if not 0 <= minimum_feature_valid_fraction <= 1:
        raise ValueError("minimum_feature_valid_fraction must lie in [0, 1]")
    if not 0 <= minimum_row_valid_fraction <= 1:
        raise ValueError("minimum_row_valid_fraction must lie in [0, 1]")
    finite_column = np.any(np.isfinite(matrix), axis=0)
    mean = np.full(matrix.shape[1], np.nan)
    scale = np.full(matrix.shape[1], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean[finite_column] = np.nanmean(matrix[:, finite_column], axis=0)
        scale[finite_column] = np.nanstd(matrix[:, finite_column], axis=0)
        standardized = (matrix - mean) / scale
    feature_mask = (
        np.mean(np.isfinite(standardized), axis=0)
        >= minimum_feature_valid_fraction
    ) & np.isfinite(scale) & (scale > 0)
    selected = standardized[:, feature_mask]
    if not selected.shape[1]:
        raise RuntimeError("no features pass the historical validity gate")
    row_mask = (
        np.mean(np.isfinite(selected), axis=1) >= minimum_row_valid_fraction
    )
    if not row_mask.any():
        raise RuntimeError("no rows pass the historical validity gate")
    filled = np.where(np.isfinite(selected[row_mask]), selected[row_mask], 0.0)
    return LegacyGeometry(
        values=filled,
        row_mask=row_mask,
        feature_mask=feature_mask,
        feature_mean=mean[feature_mask],
        feature_scale=scale[feature_mask],
        minimum_feature_valid_fraction=float(minimum_feature_valid_fraction),
        minimum_row_valid_fraction=float(minimum_row_valid_fraction),
    )


def fit_geometry_transform(
    reference: np.ndarray,
    *,
    scaling: Scaling = "center",
    minimum_valid_fraction: float = 0.95,
    variance_threshold: float = 1e-8,
) -> FrozenGeometryTransform:
    """Fit a leakage-safe transform on reference rows only."""

    matrix = np.asarray(reference, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("reference must be two-dimensional")
    if scaling not in ("center", "standard", "robust"):
        raise ValueError(f"unknown scaling {scaling!r}")
    if not 0 <= minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must lie in [0, 1]")
    valid_fraction = np.mean(np.isfinite(matrix), axis=0)
    keep = np.flatnonzero(valid_fraction >= minimum_valid_fraction)
    if not keep.size:
        raise RuntimeError("no features pass the validity gate")
    selected = matrix[:, keep]
    medians = np.nanmedian(selected, axis=0)
    finite_medians = np.isfinite(medians)
    keep = keep[finite_medians]
    selected = selected[:, finite_medians]
    medians = medians[finite_medians]
    filled = np.where(np.isfinite(selected), selected, medians)
    standard_deviation = np.std(filled, axis=0)
    varying = np.isfinite(standard_deviation) & (
        standard_deviation >= variance_threshold
    )
    keep = keep[varying]
    medians = medians[varying]
    filled = filled[:, varying]
    standard_deviation = standard_deviation[varying]
    if not keep.size:
        raise RuntimeError("no features pass the variance gate")

    if scaling == "robust":
        location = np.median(filled, axis=0)
        q25, q75 = np.quantile(filled, (0.25, 0.75), axis=0)
        robust_scale = (q75 - q25) / 1.3489795003921634
        # Do not magnify tied or nearly discrete features with a vanishing IQR.
        scale = np.where(robust_scale >= variance_threshold, robust_scale, 1.0)
    else:
        location = np.mean(filled, axis=0)
        scale = standard_deviation if scaling == "standard" else np.ones(len(keep))

    return FrozenGeometryTransform(
        keep_indices=keep,
        valid_fraction=valid_fraction[keep],
        impute_values=medians,
        location=location,
        scale=scale,
        scaling=scaling,
        minimum_valid_fraction=float(minimum_valid_fraction),
        variance_threshold=float(variance_threshold),
    )


def fit_pca_projection(
    reference: np.ndarray,
    query: np.ndarray | None = None,
    *,
    n_components: int = 50,
    random_state: int = 1729,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    """Fit PCA on reference rows and transform reference and query rows."""

    matrix = np.asarray(reference, dtype=np.float64)
    components = min(int(n_components), matrix.shape[0] - 1, matrix.shape[1])
    if components < 2:
        raise RuntimeError("at least two PCA components are required")
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=random_state,
    ).fit(matrix)
    fitted = pca.transform(matrix)
    transformed = fitted if query is None else pca.transform(np.asarray(query))
    return pca, fitted, transformed


def condensed_distances(values: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Return the non-redundant upper triangle of the pairwise distance matrix."""

    return pdist(np.asarray(values, dtype=np.float64), metric=metric)


def square_distances(condensed: np.ndarray) -> np.ndarray:
    """Expand a condensed distance vector only when a square view is needed."""

    return squareform(np.asarray(condensed, dtype=np.float64))


def distance_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if np.shape(first) != np.shape(second):
        raise ValueError("distance vectors must have identical shapes")
    return float(spearmanr(first, second).statistic)


def neighbour_overlap(
    first: np.ndarray, second: np.ndarray, *, k: int = 15
) -> float:
    """Mean overlap of the two Euclidean k-nearest-neighbour graphs."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape[0] != second.shape[0]:
        raise ValueError("representations must contain the same rows")
    if not 1 <= k < len(first):
        raise ValueError("k must lie in [1, n_rows)")
    first_nn = NearestNeighbors(n_neighbors=k + 1).fit(first).kneighbors(
        return_distance=False
    )[:, 1:]
    second_nn = NearestNeighbors(n_neighbors=k + 1).fit(second).kneighbors(
        return_distance=False
    )[:, 1:]
    return float(
        np.mean(
            [len(set(a).intersection(b)) / k for a, b in zip(first_nn, second_nn)]
        )
    )


def retrieval_summary(
    query: np.ndarray,
    query_labels: Sequence[object],
    gallery: np.ndarray,
    gallery_labels: Sequence[object],
) -> dict[str, float]:
    """Class retrieval metrics for an independent query/gallery split."""

    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)
    distances = pairwise_distances(query, gallery, metric="euclidean")
    ranking = np.argsort(distances, axis=1)
    matches = gallery_labels[ranking] == query_labels[:, None]
    relevant = np.sum(gallery_labels[None, :] == query_labels[:, None], axis=1)
    positions = np.arange(1, matches.shape[1] + 1)
    cumulative = np.cumsum(matches, axis=1)
    average_precision = np.sum((cumulative / positions) * matches, axis=1) / relevant
    return {
        "mean_average_precision": float(np.mean(average_precision)),
        "recall_at_1": float(np.mean(matches[:, 0])),
        "recall_at_5": float(np.mean(np.any(matches[:, :5], axis=1))),
    }


def nearest_neighbour_table(
    values: np.ndarray,
    names: Sequence[object],
    focal: int | str,
    *,
    k: int = 20,
) -> list[tuple[str, float]]:
    """Return nearest datasets for one observed focal dataset."""

    names = np.asarray(names).astype(str)
    if isinstance(focal, str):
        matches = np.flatnonzero(names == focal)
        if not matches.size:
            raise KeyError(f"unknown dataset {focal!r}")
        index = int(matches[0])
    else:
        index = int(focal)
    if not 1 <= k < len(names):
        raise ValueError("k must lie in [1, n_rows)")
    model = NearestNeighbors(n_neighbors=k + 1).fit(values)
    distances, indices = model.kneighbors(np.asarray(values)[[index]])
    return [
        (str(names[i]), float(distance))
        for i, distance in zip(indices[0, 1:], distances[0, 1:])
    ]
