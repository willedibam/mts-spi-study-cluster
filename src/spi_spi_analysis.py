"""Leakage-controlled preprocessing for versioned SPI--SPI feature blocks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class FrozenFeatureTransform:
    keep_indices: np.ndarray
    impute_values: np.ndarray
    center: np.ndarray
    block_scale: np.ndarray
    kept_blocks: np.ndarray
    minimum_valid_fraction: float
    variance_threshold: float
    block_balanced: bool

    def transform(self, values: np.ndarray) -> np.ndarray:
        selected = np.asarray(values, dtype=np.float64)[:, self.keep_indices]
        filled = np.where(np.isfinite(selected), selected, self.impute_values)
        return (filled - self.center) * self.block_scale


@dataclass(frozen=True)
class FrozenPC1:
    feature_transform: FrozenFeatureTransform
    component: np.ndarray
    explained_variance_ratio: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return self.feature_transform.transform(values) @ self.component


def fit_feature_transform(
    development: np.ndarray,
    feature_blocks: Sequence[str],
    *,
    minimum_valid_fraction: float = 1.0,
    variance_threshold: float = 1e-8,
    block_balanced: bool = False,
) -> FrozenFeatureTransform:
    """Fit imputation, selection, centring and optional block balancing.

    Every statistic is estimated from ``development`` only.  Block balancing
    gives each retained block unit total development variance; it does not
    whiten individual meta-features.
    """

    values = np.asarray(development, dtype=np.float64)
    blocks = np.asarray(feature_blocks, dtype=str)
    if values.ndim != 2:
        raise ValueError("development must be a two-dimensional matrix")
    if blocks.shape != (values.shape[1],):
        raise ValueError("feature_blocks length does not match feature count")
    if not 0.0 <= minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must lie in [0, 1]")
    if variance_threshold < 0:
        raise ValueError("variance_threshold must be non-negative")

    valid_fraction = np.mean(np.isfinite(values), axis=0)
    candidates = np.flatnonzero(valid_fraction >= minimum_valid_fraction)
    if candidates.size == 0:
        raise RuntimeError("no features satisfy the development validity gate")
    candidate_values = values[:, candidates]
    medians = np.nanmedian(candidate_values, axis=0)
    finite_median = np.isfinite(medians)
    candidates = candidates[finite_median]
    candidate_values = candidate_values[:, finite_median]
    medians = medians[finite_median]
    filled = np.where(np.isfinite(candidate_values), candidate_values, medians)
    variances = np.var(filled, axis=0)
    varying = np.isfinite(variances) & (np.sqrt(variances) >= variance_threshold)
    keep = candidates[varying]
    if keep.size == 0:
        raise RuntimeError("no features satisfy the development variance gate")

    filled = filled[:, varying]
    medians = medians[varying]
    kept_blocks = blocks[keep]
    center = np.mean(filled, axis=0)
    scale = np.ones(keep.size, dtype=np.float64)
    if block_balanced:
        centred = filled - center
        for block in np.unique(kept_blocks):
            member = kept_blocks == block
            total_variance = float(np.sum(np.var(centred[:, member], axis=0)))
            if total_variance <= 0 or not np.isfinite(total_variance):
                raise RuntimeError(f"retained block {block!r} has no finite variance")
            scale[member] = 1.0 / np.sqrt(total_variance)

    return FrozenFeatureTransform(
        keep_indices=keep,
        impute_values=medians,
        center=center,
        block_scale=scale,
        kept_blocks=kept_blocks,
        minimum_valid_fraction=float(minimum_valid_fraction),
        variance_threshold=float(variance_threshold),
        block_balanced=bool(block_balanced),
    )


def fit_frozen_pc1(
    development: np.ndarray,
    feature_blocks: Sequence[str],
    *,
    minimum_valid_fraction: float = 1.0,
    variance_threshold: float = 1e-8,
    block_balanced: bool = False,
) -> FrozenPC1:
    transform = fit_feature_transform(
        development,
        feature_blocks,
        minimum_valid_fraction=minimum_valid_fraction,
        variance_threshold=variance_threshold,
        block_balanced=block_balanced,
    )
    fitted = transform.transform(development)
    pca = PCA(n_components=1, svd_solver="randomized", random_state=0).fit(fitted)
    component = np.asarray(pca.components_[0], dtype=np.float64)
    anchor = int(np.argmax(np.abs(component)))
    if component[anchor] < 0:
        component = -component
    return FrozenPC1(
        feature_transform=transform,
        component=component,
        explained_variance_ratio=float(pca.explained_variance_ratio_[0]),
    )
