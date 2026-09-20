from __future__ import annotations

import numpy as np

from src.corpus_geometry import (
    condensed_distances,
    fit_geometry_transform,
    legacy_geometry,
    nearest_neighbour_table,
    neighbour_overlap,
    retrieval_summary,
    square_distances,
)


def test_legacy_geometry_reproduces_zero_fill_after_zscore() -> None:
    values = np.asarray(
        [
            [1.0, 10.0, np.nan],
            [2.0, 20.0, 4.0],
            [3.0, 30.0, 6.0],
        ]
    )
    result = legacy_geometry(
        values,
        minimum_feature_valid_fraction=0.60,
        minimum_row_valid_fraction=0.50,
    )
    assert result.values.shape == (3, 3)
    assert result.values[0, 2] == 0.0
    assert np.allclose(np.mean(result.values[:, :2], axis=0), 0.0)


def test_frozen_transform_uses_reference_statistics() -> None:
    reference = np.asarray([[0.0, 1.0], [2.0, np.nan], [4.0, 5.0]])
    query = np.asarray([[10.0, np.nan]])
    transform = fit_geometry_transform(
        reference, scaling="standard", minimum_valid_fraction=0.60
    )
    transformed = transform.transform(query)
    assert transformed.shape == (1, 2)
    assert transformed[0, 0] > 3
    assert np.isfinite(transformed).all()


def test_robust_transform_does_not_divide_by_zero_iqr() -> None:
    reference = np.asarray([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [1.0, 4.0]])
    transform = fit_geometry_transform(reference, scaling="robust")
    assert np.isfinite(transform.transform(reference)).all()
    assert np.all(transform.scale > 0)


def test_condensed_square_roundtrip_and_identical_neighbours() -> None:
    values = np.arange(24, dtype=float).reshape(8, 3)
    condensed = condensed_distances(values)
    square = square_distances(condensed)
    assert square.shape == (8, 8)
    assert np.allclose(np.diag(square), 0)
    assert neighbour_overlap(values, values.copy(), k=3) == 1.0


def test_retrieval_and_named_neighbours() -> None:
    gallery = np.asarray([[0.0], [0.1], [10.0], [10.1]])
    labels = np.asarray(["a", "a", "b", "b"])
    query = np.asarray([[0.05], [10.05]])
    summary = retrieval_summary(query, ["a", "b"], gallery, labels)
    assert summary["recall_at_1"] == 1.0
    neighbours = nearest_neighbour_table(gallery, ["a0", "a1", "b0", "b1"], "a0", k=2)
    assert neighbours[0][0] == "a1"
