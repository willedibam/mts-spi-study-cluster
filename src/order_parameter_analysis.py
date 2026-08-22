from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import hilbert
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class FrozenPC1:
    feature_indices: np.ndarray
    impute_values: np.ndarray
    center: np.ndarray
    component: np.ndarray
    explained_variance_ratio: float

    def transform(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(matrix, dtype=np.float64)[:, self.feature_indices].copy()
        missing = ~np.isfinite(values)
        if np.any(missing):
            rows, columns = np.where(missing)
            values[rows, columns] = self.impute_values[columns]
        scores = (values - self.center) @ self.component
        return scores, missing.mean(axis=1)


def fit_frozen_pc1(
    matrix: np.ndarray,
    *,
    variance_threshold: float = 0.05,
) -> FrozenPC1:
    """Fit target-free PC1, retaining only fully finite development features."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("matrix must contain at least two rows")
    keep = np.isfinite(values).all(axis=0) & (
        np.std(values, axis=0) >= float(variance_threshold)
    )
    if not np.any(keep):
        raise RuntimeError("no finite varying development features remain")
    feature_indices = np.flatnonzero(keep)
    selected = values[:, feature_indices]
    pca = PCA(n_components=1, svd_solver="full").fit(selected)
    component = pca.components_[0].copy()
    pivot = int(np.argmax(np.abs(component)))
    if component[pivot] < 0.0:
        component *= -1.0
    return FrozenPC1(
        feature_indices=feature_indices,
        impute_values=np.median(selected, axis=0),
        center=pca.mean_.copy(),
        component=component,
        explained_variance_ratio=float(pca.explained_variance_ratio_[0]),
    )


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if valid.sum() < 3:
        return float("nan")
    x_values = x_values[valid]
    y_values = y_values[valid]
    if np.unique(x_values).size < 2 or np.unique(y_values).size < 2:
        return float("nan")
    return float(spearmanr(x_values, y_values).statistic)


def residualize_by_group(values: Sequence[float], groups: Sequence[object]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    groups_array = np.asarray(groups)
    if values_array.shape[0] != groups_array.shape[0]:
        raise ValueError("values and groups must have the same length")
    residuals = np.full(values_array.shape, np.nan, dtype=np.float64)
    for group in np.unique(groups_array):
        mask = groups_array == group
        residuals[mask] = values_array[mask] - np.nanmean(values_array[mask])
    return residuals


def input_only_features(timeseries: np.ndarray) -> dict[str, np.ndarray | float]:
    """Simple fair baselines computed only from the supplied scalar MTS."""

    data = np.asarray(timeseries, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 4 or data.shape[1] < 2:
        raise ValueError("timeseries must have shape (T>=4, M>=2)")
    correlation = np.corrcoef(data, rowvar=False)
    upper = correlation[np.triu_indices(data.shape[1], k=1)]
    covariance = np.cov(data, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    trace = float(np.sum(eigenvalues))
    analytic_phase = np.angle(hilbert(data - data.mean(axis=0), axis=0))
    phase_coherence = np.abs(np.mean(np.exp(1j * analytic_phase), axis=1))
    power = np.abs(np.fft.rfft(data - data.mean(axis=0), axis=0)) ** 2
    power = power[1:]
    probabilities = power / np.maximum(power.sum(axis=0, keepdims=True), 1e-15)
    log_probabilities = np.zeros_like(probabilities)
    positive = probabilities > 0.0
    log_probabilities[positive] = np.log(probabilities[positive])
    spectral_entropy = -np.sum(probabilities * log_probabilities, axis=0)
    spectral_entropy /= np.log(max(2, probabilities.shape[0]))
    return {
        "correlation_vector": upper,
        "mean_abs_correlation": float(np.nanmean(np.abs(upper))),
        "covariance_leading_fraction": (
            float(eigenvalues[-1] / trace) if trace > 0.0 else float("nan")
        ),
        "analytic_phase_coherence": float(np.nanmean(phase_coherence)),
        "mean_temporal_spectral_entropy": float(np.mean(spectral_entropy)),
        "pooled_mean": float(np.mean(data)),
        "pooled_std": float(np.std(data)),
    }


def clustered_bootstrap_difference(
    truth: Sequence[float],
    prediction_a: Sequence[float],
    prediction_b: Sequence[float],
    clusters: Sequence[object],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Cluster-bootstrap MAE(a)-MAE(b); negative values favour prediction a."""

    truth_array = np.asarray(truth, dtype=np.float64)
    a_array = np.asarray(prediction_a, dtype=np.float64)
    b_array = np.asarray(prediction_b, dtype=np.float64)
    cluster_array = np.asarray(clusters)
    if not (
        truth_array.shape == a_array.shape == b_array.shape == cluster_array.shape
    ):
        raise ValueError("truth, predictions and clusters must have matching shapes")
    unique = np.unique(cluster_array)
    if unique.size < 2:
        raise ValueError("at least two clusters are required")
    indices = {cluster: np.flatnonzero(cluster_array == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    differences = np.empty(int(n_resamples), dtype=np.float64)
    for draw in range(int(n_resamples)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        selected = np.concatenate([indices[cluster] for cluster in sampled])
        differences[draw] = np.mean(np.abs(truth_array[selected] - a_array[selected])) - np.mean(
            np.abs(truth_array[selected] - b_array[selected])
        )
    return differences


def clustered_bootstrap_mae(
    truth: Sequence[float],
    prediction: Sequence[float],
    clusters: Sequence[object],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    cluster_array = np.asarray(clusters)
    if not (truth_array.shape == prediction_array.shape == cluster_array.shape):
        raise ValueError("truth, prediction and clusters must have matching shapes")
    unique = np.unique(cluster_array)
    if unique.size < 2:
        raise ValueError("at least two clusters are required")
    indices = {cluster: np.flatnonzero(cluster_array == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    values = np.empty(int(n_resamples), dtype=np.float64)
    for draw in range(int(n_resamples)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        selected = np.concatenate([indices[cluster] for cluster in sampled])
        values[draw] = np.mean(
            np.abs(truth_array[selected] - prediction_array[selected])
        )
    return values


def clustered_bootstrap_spearman(
    coordinate: Sequence[float],
    truth: Sequence[float],
    groups: Sequence[object],
    clusters: Sequence[object],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster-bootstrap overall and within-group Spearman associations."""

    coordinate_array = np.asarray(coordinate, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    group_array = np.asarray(groups)
    cluster_array = np.asarray(clusters)
    if not (
        coordinate_array.shape
        == truth_array.shape
        == group_array.shape
        == cluster_array.shape
    ):
        raise ValueError("all inputs must have matching shapes")
    unique = np.unique(cluster_array)
    if unique.size < 2:
        raise ValueError("at least two clusters are required")
    indices = {cluster: np.flatnonzero(cluster_array == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    overall = np.empty(int(n_resamples), dtype=np.float64)
    within = np.empty(int(n_resamples), dtype=np.float64)
    for draw in range(int(n_resamples)):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        selected = np.concatenate([indices[cluster] for cluster in sampled])
        overall[draw] = safe_spearman(
            coordinate_array[selected], truth_array[selected]
        )
        within[draw] = safe_spearman(
            residualize_by_group(coordinate_array[selected], group_array[selected]),
            residualize_by_group(truth_array[selected], group_array[selected]),
        )
    return overall, within
