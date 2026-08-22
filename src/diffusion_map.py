from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class FrozenDiffusionMap:
    pca_mean: np.ndarray
    pca_components: np.ndarray
    reference_scores: np.ndarray
    reference_density: np.ndarray
    reference_eigenfunction: np.ndarray
    eigenvalue: float
    bandwidth: float
    neighbours: int
    explained_variance: float

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        scores = (values - self.pca_mean) @ self.pca_components.T
        distances = cdist(scores, self.reference_scores, metric="sqeuclidean")
        weights = np.exp(-distances / self.bandwidth)
        density = np.maximum(weights.sum(axis=1), np.finfo(float).tiny)
        kernel = weights / (density[:, None] * self.reference_density[None, :])
        degree = np.maximum(kernel.sum(axis=1), np.finfo(float).tiny)
        transition = kernel / degree[:, None]
        return (transition @ self.reference_eigenfunction) / self.eigenvalue


def fit_diffusion_map(
    matrix: np.ndarray,
    *,
    variance_target: float = 0.90,
    max_components: int = 20,
    random_state: int = 0,
) -> tuple[FrozenDiffusionMap, np.ndarray]:
    """Fit the first nontrivial alpha=1 diffusion-map coordinate.

    The input must already be finite and use frozen feature scaling. PCA is
    unwhitened; its dimension is the first component count reaching the
    variance target, capped at ``max_components``.
    """

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 5 or not np.isfinite(values).all():
        raise ValueError("matrix must be a finite 2-D array with at least five rows")
    maximum = min(int(max_components), values.shape[0] - 1, values.shape[1])
    pca = PCA(
        n_components=maximum,
        svd_solver="randomized",
        iterated_power=7,
        random_state=int(random_state),
    ).fit(values)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    reached = np.flatnonzero(cumulative >= float(variance_target))
    dimension = int(reached[0] + 1) if reached.size else maximum
    components = pca.components_[:dimension].copy()
    scores = (values - pca.mean_) @ components.T

    squared = cdist(scores, scores, metric="sqeuclidean")
    neighbours = max(2, int(np.floor(np.sqrt(values.shape[0]))))
    kth = np.partition(squared, neighbours, axis=1)[:, neighbours]
    bandwidth = float(np.median(kth))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise RuntimeError("diffusion-map bandwidth is not positive")
    weights = np.exp(-squared / bandwidth)
    np.fill_diagonal(weights, 0.0)
    density = np.maximum(weights.sum(axis=1), np.finfo(float).tiny)
    kernel = weights / (density[:, None] * density[None, :])
    degree = np.maximum(kernel.sum(axis=1), np.finfo(float).tiny)
    symmetric = kernel / np.sqrt(degree[:, None] * degree[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalue = float(eigenvalues[order[1]])
    eigenfunction = eigenvectors[:, order[1]] / np.sqrt(degree)
    pivot = int(np.argmax(np.abs(eigenfunction)))
    if eigenfunction[pivot] < 0.0:
        eigenfunction *= -1.0
    model = FrozenDiffusionMap(
        pca_mean=pca.mean_.copy(),
        pca_components=components,
        reference_scores=scores,
        reference_density=density,
        reference_eigenfunction=eigenfunction,
        eigenvalue=eigenvalue,
        bandwidth=bandwidth,
        neighbours=neighbours,
        explained_variance=float(cumulative[dimension - 1]),
    )
    return model, eigenfunction.copy()
