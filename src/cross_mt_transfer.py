"""Leakage-controlled utilities for the frozen cross-M,T transfer study."""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.special import xlogy
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances

from .spi_spi_analysis import FrozenFeatureTransform, fit_feature_transform


SAMPLE_FIELDS = ("y", "dataset_paths", "M", "T", "instance")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_artifact(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}
    if str(payload.get("feature_contract", np.asarray("")).item()) != "direction_preserving_v2":
        raise ValueError(f"not a direction_preserving_v2 artifact: {path}")
    return payload


def combine_feature_artifacts(
    artifacts: Sequence[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not artifacts:
        raise ValueError("artifacts is empty")
    scalar_identity = (
        "feature_contract",
        "metric",
        "schema_sha256",
        "sym_schema_sha256",
        "dir_schema_sha256",
    )
    array_identity = (
        "spi_order",
        "directed_flags",
        "feature_block",
        "feature_relation",
        "feature_spi_a",
        "feature_spi_b",
    )
    reference = artifacts[0]
    for artifact in artifacts[1:]:
        for key in scalar_identity:
            if str(artifact[key].item()) != str(reference[key].item()):
                raise ValueError(f"artifact identity mismatch for {key}")
        for key in array_identity:
            if not np.array_equal(artifact[key], reference[key]):
                raise ValueError(f"artifact schema mismatch for {key}")
    combined = {
        key: np.concatenate([artifact[key] for artifact in artifacts], axis=0)
        for key in (*SAMPLE_FIELDS, "X_sym", "X_dir")
    }
    for key in (*scalar_identity, *array_identity):
        combined[key] = reference[key]
    return combined


def feature_view(
    artifact: dict[str, np.ndarray], name: str
) -> tuple[np.ndarray, np.ndarray, bool]:
    if name == "sym":
        values = np.asarray(artifact["X_sym"], dtype=np.float64)
        return values, np.repeat("sym", values.shape[1]), False
    if name == "dir":
        values = np.asarray(artifact["X_dir"], dtype=np.float64)
        return values, np.repeat("dir", values.shape[1]), False
    if name == "augmented_balanced":
        sym = np.asarray(artifact["X_sym"], dtype=np.float64)
        direction = np.asarray(artifact["X_dir"], dtype=np.float64)
        return (
            np.concatenate((sym, direction), axis=1),
            np.concatenate((np.repeat("sym", sym.shape[1]), np.repeat("dir", direction.shape[1]))),
            True,
        )
    raise ValueError(f"unknown representation {name!r}")


@dataclass(frozen=True)
class FrozenProjection:
    feature_transform: FrozenFeatureTransform
    standard_scale: np.ndarray
    pca: PCA

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = self.feature_transform.transform(values) / self.standard_scale
        return self.pca.transform(transformed)


@dataclass(frozen=True)
class FrozenCellModel:
    held_cell: str
    projection: FrozenProjection
    classifier: LogisticRegression
    gallery_coordinates: np.ndarray
    gallery_labels: np.ndarray
    gallery_M: np.ndarray
    gallery_T: np.ndarray


@dataclass(frozen=True)
class FrozenSharedModel:
    projection: FrozenProjection
    size_classifiers: dict[str, dict[str, LogisticRegression]]


def fit_projection(
    development: np.ndarray,
    feature_blocks: Sequence[str],
    *,
    minimum_valid_fraction: float,
    variance_threshold: float,
    block_balanced: bool,
    standardize: bool,
    dimensions: int,
    random_state: int,
) -> tuple[FrozenProjection, np.ndarray]:
    transform = fit_feature_transform(
        development,
        feature_blocks,
        minimum_valid_fraction=minimum_valid_fraction,
        variance_threshold=variance_threshold,
        block_balanced=block_balanced,
    )
    transformed = transform.transform(development)
    scale = np.std(transformed, axis=0) if standardize else np.ones(transformed.shape[1])
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    transformed = transformed / scale
    n_components = min(dimensions, transformed.shape[0] - 1, transformed.shape[1])
    if n_components < 1:
        raise RuntimeError("insufficient rows/features for PCA")
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=random_state,
    ).fit(transformed)
    projection = FrozenProjection(transform, scale, pca)
    return projection, pca.transform(transformed)


def _safe_autocorrelation(values: np.ndarray, lag: int) -> np.ndarray:
    if values.shape[0] <= lag:
        return np.full(values.shape[1], np.nan)
    left = values[:-lag] - np.mean(values[:-lag], axis=0)
    right = values[lag:] - np.mean(values[lag:], axis=0)
    denominator = np.sqrt(np.sum(left**2, axis=0) * np.sum(right**2, axis=0))
    return np.divide(
        np.sum(left * right, axis=0),
        denominator,
        out=np.full(values.shape[1], np.nan),
        where=denominator > 0,
    )


def _spectral_entropy(values: np.ndarray) -> np.ndarray:
    centred = values - np.mean(values, axis=0)
    power = np.abs(np.fft.rfft(centred, axis=0)) ** 2
    power = power[1:]
    total = np.sum(power, axis=0)
    probability = np.divide(power, total, out=np.zeros_like(power), where=total > 0)
    entropy = -np.sum(xlogy(probability, probability), axis=0)
    denominator = np.log(max(2, power.shape[0]))
    return np.divide(entropy, denominator, out=np.full(values.shape[1], np.nan), where=total > 0)


def _pool_channels(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    summaries = (
        ("mean", lambda x: np.nanmean(x)),
        ("std", lambda x: np.nanstd(x)),
        ("q10", lambda x: np.nanquantile(x, 0.10)),
        ("q50", lambda x: np.nanquantile(x, 0.50)),
        ("q90", lambda x: np.nanquantile(x, 0.90)),
    )
    pooled: list[float] = []
    names: list[str] = []
    for descriptor_index in range(values.shape[0]):
        row = values[descriptor_index]
        for summary_name, function in summaries:
            pooled.append(float(function(row)) if np.isfinite(row).any() else np.nan)
            names.append(f"d{descriptor_index}_{summary_name}")
    return np.asarray(pooled, dtype=np.float64), names


def pooled_univariate_features(timeseries: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(timeseries, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("timeseries must have shape (T, M) with T,M >= 2")
    quantiles = np.quantile(values, (0.25, 0.5, 0.75), axis=0)
    iqr = quantiles[2] - quantiles[0]
    robust_skew = np.divide(
        quantiles[2] + quantiles[0] - 2 * quantiles[1],
        iqr,
        out=np.zeros_like(iqr),
        where=iqr > 0,
    )
    standard_deviation = np.std(values, axis=0)
    difference_ratio = np.divide(
        np.std(np.diff(values, axis=0), axis=0),
        standard_deviation,
        out=np.full(values.shape[1], np.nan),
        where=standard_deviation > 0,
    )
    median_crossing = np.mean(
        np.signbit(values[1:] - quantiles[1]) != np.signbit(values[:-1] - quantiles[1]),
        axis=0,
    )
    descriptors = np.vstack(
        (
            np.mean(values, axis=0),
            standard_deviation,
            quantiles[1],
            iqr,
            robust_skew,
            _safe_autocorrelation(values, 1),
            _safe_autocorrelation(values, 5),
            difference_ratio,
            _spectral_entropy(values),
            median_crossing,
        )
    )
    pooled, suffixes = _pool_channels(descriptors)
    descriptor_names = (
        "mean",
        "std",
        "median",
        "iqr",
        "robust_skew",
        "autocorr_1",
        "autocorr_5",
        "difference_ratio",
        "spectral_entropy",
        "median_crossing",
    )
    names = [f"{descriptor_names[int(name.split('_')[0][1:])]}_{name.split('_', 1)[1]}" for name in suffixes]
    return pooled, names


def _distribution_summary(values: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    names = [f"{prefix}_{name}" for name in ("mean", "std", "q10", "q25", "q50", "q75", "q90")]
    if finite.size == 0:
        return [np.nan] * len(names), names
    result = [float(np.mean(finite)), float(np.std(finite))]
    result.extend(float(value) for value in np.quantile(finite, (0.10, 0.25, 0.50, 0.75, 0.90)))
    return result, names


def pooled_dependence_features(timeseries: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(timeseries, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("timeseries must have at least two channels")
    centred = values - np.mean(values, axis=0)
    scale = np.sqrt(np.sum(centred**2, axis=0))
    normalized = np.divide(centred, scale, out=np.zeros_like(centred), where=scale > 0)
    correlation = normalized.T @ normalized
    upper = correlation[np.triu_indices(values.shape[1], k=1)]

    left = values[:-1] - np.mean(values[:-1], axis=0)
    right = values[1:] - np.mean(values[1:], axis=0)
    left_scale = np.sqrt(np.sum(left**2, axis=0))
    right_scale = np.sqrt(np.sum(right**2, axis=0))
    left = np.divide(left, left_scale, out=np.zeros_like(left), where=left_scale > 0)
    right = np.divide(right, right_scale, out=np.zeros_like(right), where=right_scale > 0)
    lagged = left.T @ right
    ordered = lagged[~np.eye(lagged.shape[0], dtype=bool)]

    result: list[float] = []
    names: list[str] = []
    for vector, prefix in (
        (upper, "correlation"),
        (np.abs(upper), "absolute_correlation"),
        (ordered, "lag1_ordered"),
        (np.abs(ordered), "absolute_lag1_ordered"),
    ):
        block, block_names = _distribution_summary(vector, prefix)
        result.extend(block)
        names.extend(block_names)

    finite_correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(finite_correlation, 1.0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(finite_correlation), 0.0)
    eigen_probability = eigenvalues / max(float(np.sum(eigenvalues)), np.finfo(float).eps)
    positive = eigen_probability > 0
    effective_rank = float(np.exp(-np.sum(eigen_probability[positive] * np.log(eigen_probability[positive]))))
    participation = float(1.0 / np.sum(eigen_probability**2))
    extras = (
        float(np.max(eigen_probability)),
        effective_rank / values.shape[1],
        participation / values.shape[1],
        float(np.mean(np.abs(upper))),
    )
    result.extend(extras)
    names.extend(("top_correlation_eigen_fraction", "normalized_effective_rank", "normalized_participation_ratio", "mean_absolute_correlation"))
    return np.asarray(result, dtype=np.float64), names


def pooled_baseline_features(timeseries: np.ndarray) -> dict[str, tuple[np.ndarray, list[str]]]:
    univariate = pooled_univariate_features(timeseries)
    dependence = pooled_dependence_features(timeseries)
    return {
        "pooled_univariate": univariate,
        "pooled_dependence": dependence,
        "pooled_combined": (
            np.concatenate((univariate[0], dependence[0])),
            [f"univariate:{name}" for name in univariate[1]]
            + [f"dependence:{name}" for name in dependence[1]],
        ),
    }


def _baseline_row(path: str) -> tuple[dict[str, np.ndarray], dict[str, list[str]], str]:
    timeseries_path = Path(path) / "timeseries.npy"
    features = pooled_baseline_features(np.load(timeseries_path))
    return (
        {name: value[0] for name, value in features.items()},
        {name: value[1] for name, value in features.items()},
        file_sha256(timeseries_path),
    )


def build_pooled_baseline_matrices(
    dataset_paths: Sequence[str], *, workers: int = 1
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], list[str]]:
    paths = [str(path) for path in dataset_paths]
    if workers == 1:
        rows = [_baseline_row(path) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_baseline_row, paths))
    names = rows[0][1]
    for _, current_names, _ in rows[1:]:
        if current_names != names:
            raise RuntimeError("pooled baseline schema changed between rows")
    matrices = {
        name: np.vstack([row[0][name] for row in rows]).astype(np.float32)
        for name in names
    }
    return matrices, names, [row[2] for row in rows]


def fit_logistic_classifier(
    coordinates: np.ndarray,
    labels: np.ndarray,
    *,
    C: float,
    solver: str,
    max_iter: int,
    tolerance: float,
    random_state: int,
) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        tol=tolerance,
        random_state=random_state,
    ).fit(coordinates, labels)


def retrieval_scores(
    query: np.ndarray,
    query_labels: np.ndarray,
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    distances = pairwise_distances(query, gallery, metric="euclidean")
    ranking = np.argsort(distances, axis=1)
    ranked_labels = gallery_labels[ranking]
    matches = ranked_labels == query_labels[:, None]
    cumulative = np.cumsum(matches, axis=1)
    positions = np.arange(1, matches.shape[1] + 1)
    relevant = np.sum(gallery_labels[None, :] == query_labels[:, None], axis=1)
    average_precision = np.sum((cumulative / positions) * matches, axis=1) / relevant
    return {
        "average_precision": average_precision,
        "recall_at_1": matches[:, 0].astype(float),
        "recall_at_5": np.any(matches[:, :5], axis=1).astype(float),
    }


def geometry_metrics(coordinates: np.ndarray, labels: np.ndarray, cells: np.ndarray) -> dict[str, float]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    labels = np.asarray(labels)
    cells = np.asarray(cells)
    grand = np.mean(coordinates, axis=0)
    total = float(np.sum((coordinates - grand) ** 2))
    class_ss = 0.0
    for label in np.unique(labels):
        member = labels == label
        class_ss += float(np.sum(member)) * float(np.sum((np.mean(coordinates[member], axis=0) - grand) ** 2))
    cell_ss = 0.0
    for cell in np.unique(cells):
        member = cells == cell
        cell_ss += float(np.sum(member)) * float(np.sum((np.mean(coordinates[member], axis=0) - grand) ** 2))
    class_eta = class_ss / total if total > 0 else np.nan
    cell_eta = cell_ss / total if total > 0 else np.nan

    distances = pairwise_distances(coordinates, metric="euclidean")
    upper = np.triu(np.ones(distances.shape, dtype=bool), k=1)
    same_class_cross_cell = upper & (labels[:, None] == labels[None, :]) & (cells[:, None] != cells[None, :])
    different_class_same_cell = upper & (labels[:, None] != labels[None, :]) & (cells[:, None] == cells[None, :])
    numerator = float(np.median(distances[same_class_cross_cell]))
    denominator = float(np.median(distances[different_class_same_cell]))
    return {
        "class_eta_squared": class_eta,
        "cell_eta_squared": cell_eta,
        "class_to_cell_ratio": class_eta / cell_eta if cell_eta > 0 else np.inf,
        "matched_distance_ratio": numerator / denominator if denominator > 0 else np.inf,
    }


def stratified_bootstrap_interval(
    metric: Callable[[np.ndarray], float],
    strata: np.ndarray,
    *,
    repetitions: int,
    confidence_level: float,
    random_state: int,
) -> tuple[float, float]:
    strata = np.asarray(strata)
    rng = np.random.default_rng(random_state)
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    estimates = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        sampled = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        estimates[repetition] = metric(sampled)
    alpha = (1.0 - confidence_level) / 2.0
    return tuple(float(value) for value in np.quantile(estimates, (alpha, 1.0 - alpha)))


def held_label_permutation_test(
    metric: Callable[[np.ndarray], float],
    labels: np.ndarray,
    cells: np.ndarray,
    observed: float,
    *,
    repetitions: int,
    random_state: int,
) -> tuple[float, float]:
    labels = np.asarray(labels)
    cells = np.asarray(cells)
    rng = np.random.default_rng(random_state)
    groups = [np.flatnonzero(cells == value) for value in np.unique(cells)]
    null = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        permuted = labels.copy()
        for group in groups:
            permuted[group] = rng.permutation(permuted[group])
        null[repetition] = metric(permuted)
    p_value = (1.0 + float(np.sum(null >= observed))) / (repetitions + 1.0)
    return float(np.mean(null)), p_value
