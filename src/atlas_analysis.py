"""Unsupervised analysis utilities for the unified SPI--SPI corpus atlas."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class AtlasTransform:
    """Frozen missingness gate, median imputation and mean centring."""

    keep_indices: np.ndarray
    valid_fraction: np.ndarray
    impute_values: np.ndarray
    center: np.ndarray
    minimum_valid_fraction: float
    variance_threshold: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        selected = np.asarray(values, dtype=np.float64)[:, self.keep_indices]
        filled = np.where(np.isfinite(selected), selected, self.impute_values)
        return filled - self.center


def load_unified_artifact(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}
    contract = str(np.asarray(payload.get("feature_contract", "")).item())
    if contract != "unified_ordered_v3":
        raise ValueError(f"not a unified_ordered_v3 artifact: {path}")
    values = np.asarray(payload.get("X"))
    validity = np.asarray(payload.get("validity_mask"))
    if values.ndim != 2 or validity.shape != values.shape:
        raise ValueError("feature matrix and validity mask are inconsistent")
    if not np.array_equal(validity, np.isfinite(values)):
        raise ValueError("validity mask does not match finite feature values")
    n_spis = len(payload["spi_order"])
    if values.shape[1] != n_spis * (n_spis - 1) // 2:
        raise ValueError("unified artifact does not contain K choose 2 features")
    return payload


def fit_atlas_transform(
    values: np.ndarray,
    *,
    minimum_valid_fraction: float = 0.95,
    variance_threshold: float = 1e-8,
) -> AtlasTransform:
    """Fit the primary covariance-PCA preprocessing without feature whitening."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("values must be two-dimensional")
    if not 0 <= minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must lie in [0, 1]")
    valid_fraction = np.mean(np.isfinite(matrix), axis=0)
    candidates = np.flatnonzero(valid_fraction >= minimum_valid_fraction)
    if not candidates.size:
        raise RuntimeError("no features pass the validity gate")
    selected = matrix[:, candidates]
    medians = np.nanmedian(selected, axis=0)
    finite_medians = np.isfinite(medians)
    candidates = candidates[finite_medians]
    selected = selected[:, finite_medians]
    medians = medians[finite_medians]
    filled = np.where(np.isfinite(selected), selected, medians)
    standard_deviation = np.std(filled, axis=0)
    varying = np.isfinite(standard_deviation) & (
        standard_deviation >= variance_threshold
    )
    candidates = candidates[varying]
    medians = medians[varying]
    filled = filled[:, varying]
    if not candidates.size:
        raise RuntimeError("no features pass the variance gate")
    return AtlasTransform(
        keep_indices=candidates,
        valid_fraction=valid_fraction[candidates],
        impute_values=medians,
        center=np.mean(filled, axis=0),
        minimum_valid_fraction=float(minimum_valid_fraction),
        variance_threshold=float(variance_threshold),
    )


def fit_atlas_pca(
    transformed: np.ndarray,
    *,
    n_components: int = 100,
    random_state: int = 1729,
) -> tuple[PCA, np.ndarray]:
    values = np.asarray(transformed, dtype=np.float64)
    components = min(n_components, values.shape[0] - 1, values.shape[1])
    if components < 2:
        raise RuntimeError("at least two PCA components are required")
    pca = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=random_state,
    )
    return pca, pca.fit_transform(values)


def neighbour_recall(reference: np.ndarray, embedding: np.ndarray, k: int) -> float:
    """Mean fraction of each point's reference neighbours retained."""

    if k < 1 or k >= len(reference):
        raise ValueError("k must lie between 1 and n_samples - 1")
    reference_indices = NearestNeighbors(n_neighbors=k + 1).fit(reference).kneighbors(
        return_distance=False
    )[:, 1:]
    embedding_indices = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(
        return_distance=False
    )[:, 1:]
    return float(
        np.mean(
            [
                len(set(first).intersection(second)) / k
                for first, second in zip(reference_indices, embedding_indices)
            ]
        )
    )


def embedding_quality(
    reference: np.ndarray,
    embedding: np.ndarray,
    *,
    neighbours: Sequence[int] = (15, 30),
) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in neighbours:
        if k >= len(reference):
            continue
        if k < len(reference) / 2:
            result[f"trustworthiness_{k}"] = float(
                trustworthiness(reference, embedding, n_neighbors=k)
            )
        result[f"neighbour_recall_{k}"] = neighbour_recall(reference, embedding, k)
    return result


def procrustes_stability(embeddings: Sequence[np.ndarray]) -> float:
    """Return one minus mean Procrustes disparity across replicate embeddings."""

    if len(embeddings) < 2:
        return np.nan
    disparities = [
        procrustes(first, second)[2]
        for first, second in combinations(embeddings, 2)
    ]
    return float(1.0 - np.mean(disparities))


def predictive_subsample_stability(
    values: np.ndarray,
    factory: Callable[[int], object],
    *,
    seeds: Sequence[int],
    fraction: float = 0.8,
) -> tuple[float, list[np.ndarray]]:
    """Fit on repeated subsamples, predict every row, and compare partitions."""

    predictions: list[np.ndarray] = []
    sample_size = max(2, int(round(fraction * len(values))))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        subset = rng.choice(len(values), size=sample_size, replace=False)
        model = factory(seed)
        model.fit(values[subset])  # type: ignore[attr-defined]
        predictions.append(np.asarray(model.predict(values), dtype=int))  # type: ignore[attr-defined]
    scores = [
        adjusted_rand_score(first, second)
        for first, second in combinations(predictions, 2)
    ]
    return float(np.mean(scores)), predictions


def density_subsample_stability(
    values: np.ndarray,
    factory: Callable[[], object],
    *,
    seeds: Sequence[int],
    fraction: float = 0.8,
) -> float:
    """Compare density partitions on shared, non-noise subsample members."""

    sample_size = max(2, int(round(fraction * len(values))))
    assignments: list[np.ndarray] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        subset = np.sort(rng.choice(len(values), size=sample_size, replace=False))
        labels = np.asarray(factory().fit_predict(values[subset]), dtype=int)  # type: ignore[attr-defined]
        assignment = np.full(len(values), -2, dtype=int)
        assignment[subset] = labels
        assignments.append(assignment)
    scores: list[float] = []
    for first, second in combinations(assignments, 2):
        common = (first >= 0) & (second >= 0)
        if common.sum() < 20:
            continue
        if len(np.unique(first[common])) < 2 or len(np.unique(second[common])) < 2:
            continue
        scores.append(adjusted_rand_score(first[common], second[common]))
    return float(np.mean(scores)) if scores else np.nan


def gmm_grid(
    pca_scores: np.ndarray,
    *,
    dimensions: Sequence[int],
    components: Sequence[int],
    covariance_types: Sequence[str] = ("diag", "tied", "full"),
    random_state: int = 1729,
) -> tuple[pd.DataFrame, dict[tuple[int, int, str], GaussianMixture]]:
    """Fit a bounded GMM grid; BIC selects within the declared search space."""

    rows: list[dict[str, float | int | str]] = []
    models: dict[tuple[int, int, str], GaussianMixture] = {}
    for dimension in dimensions:
        values = pca_scores[:, :dimension]
        for covariance_type in covariance_types:
            if covariance_type == "full" and dimension > 20:
                continue
            for n_components in components:
                model = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    reg_covar=1e-6,
                    n_init=3,
                    max_iter=500,
                    random_state=random_state,
                ).fit(values)
                labels = model.predict(values)
                counts = np.bincount(labels, minlength=n_components)
                probability = model.predict_proba(values)
                normalized_entropy = (
                    -np.mean(np.sum(probability * np.log(probability + 1e-15), axis=1))
                    / np.log(n_components)
                    if n_components > 1
                    else 0.0
                )
                rows.append(
                    {
                        "method": "gmm",
                        "dimensions": dimension,
                        "clusters": n_components,
                        "covariance_type": covariance_type,
                        "bic": float(model.bic(values)),
                        "aic": float(model.aic(values)),
                        "minimum_cluster_size": int(counts.min()),
                        "mean_max_posterior": float(np.mean(probability.max(axis=1))),
                        "normalized_entropy": float(normalized_entropy),
                    }
                )
                models[(dimension, n_components, covariance_type)] = model
    return pd.DataFrame(rows), models


def kmeans_grid(
    pca_scores: np.ndarray,
    *,
    dimensions: Sequence[int],
    clusters: Sequence[int],
    random_state: int = 1729,
) -> tuple[pd.DataFrame, dict[tuple[int, int], KMeans]]:
    rows: list[dict[str, float | int | str]] = []
    models: dict[tuple[int, int], KMeans] = {}
    for dimension in dimensions:
        values = pca_scores[:, :dimension]
        for n_clusters in clusters:
            model = KMeans(
                n_clusters=n_clusters,
                n_init=30,
                random_state=random_state,
            ).fit(values)
            labels = model.labels_
            counts = np.bincount(labels, minlength=n_clusters)
            rows.append(
                {
                    "method": "kmeans",
                    "dimensions": dimension,
                    "clusters": n_clusters,
                    "silhouette": float(silhouette_score(values, labels)),
                    "davies_bouldin": float(davies_bouldin_score(values, labels)),
                    "inertia": float(model.inertia_),
                    "minimum_cluster_size": int(counts.min()),
                }
            )
            models[(dimension, n_clusters)] = model
    return pd.DataFrame(rows), models


def hdbscan_grid(
    pca_scores: np.ndarray,
    *,
    dimensions: Sequence[int],
    minimum_cluster_sizes: Sequence[int],
    minimum_samples: Sequence[int],
) -> tuple[pd.DataFrame, dict[tuple[int, int, int], HDBSCAN]]:
    rows: list[dict[str, float | int | str]] = []
    models: dict[tuple[int, int, int], HDBSCAN] = {}
    for dimension in dimensions:
        values = pca_scores[:, :dimension]
        for min_cluster_size in minimum_cluster_sizes:
            for min_samples in minimum_samples:
                model = HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    cluster_selection_method="eom",
                    allow_single_cluster=False,
                ).fit(values)
                labels = model.labels_
                retained = labels >= 0
                unique = np.unique(labels[retained])
                silhouette = (
                    float(silhouette_score(values[retained], labels[retained]))
                    if len(unique) >= 2 and retained.sum() > len(unique)
                    else np.nan
                )
                rows.append(
                    {
                        "method": "hdbscan",
                        "dimensions": dimension,
                        "clusters": len(unique),
                        "min_cluster_size": min_cluster_size,
                        "min_samples": min_samples,
                        "coverage": float(np.mean(retained)),
                        "silhouette_nonnoise": silhouette,
                        "selection_score": (
                            silhouette * np.mean(retained)
                            if np.isfinite(silhouette)
                            else np.nan
                        ),
                    }
                )
                models[(dimension, min_cluster_size, min_samples)] = model
    return pd.DataFrame(rows), models


def cluster_medoids(values: np.ndarray, labels: np.ndarray) -> dict[int, int]:
    """Return the observed member nearest its cluster's members in total distance."""

    result: dict[int, int] = {}
    for label in np.unique(labels):
        if label < 0:
            continue
        members = np.flatnonzero(labels == label)
        cluster = values[members]
        squared = np.sum(cluster**2, axis=1)
        distances = np.sqrt(
            np.maximum(
                squared[:, None] + squared[None, :] - 2 * cluster @ cluster.T,
                0,
            )
        )
        result[int(label)] = int(members[np.argmin(distances.sum(axis=1))])
    return result
