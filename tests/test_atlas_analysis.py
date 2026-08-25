from pathlib import Path

import numpy as np

from src.atlas_analysis import (
    cluster_medoids,
    density_subsample_stability,
    fit_atlas_pca,
    fit_atlas_transform,
    embedding_quality,
    load_unified_artifact,
    neighbour_recall,
    predictive_subsample_stability,
)
from sklearn.cluster import KMeans


def test_atlas_transform_gates_imputes_and_centres() -> None:
    values = np.array(
        [
            [1.0, 2.0, np.nan, 7.0],
            [2.0, 2.0, 4.0, np.nan],
            [3.0, 2.0, 6.0, np.nan],
            [4.0, 2.0, 8.0, np.nan],
        ]
    )
    transform = fit_atlas_transform(
        values,
        minimum_valid_fraction=0.75,
        variance_threshold=1e-8,
    )
    # Constant column 1 and low-validity column 3 are removed.
    np.testing.assert_array_equal(transform.keep_indices, [0, 2])
    transformed = transform.transform(values)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-12)
    assert np.isfinite(transformed).all()


def test_unified_artifact_loader_enforces_k_choose_two(tmp_path: Path) -> None:
    path = tmp_path / "features.npz"
    np.savez(
        path,
        feature_contract="unified_ordered_v3",
        X=np.ones((2, 3)),
        validity_mask=np.ones((2, 3), dtype=bool),
        spi_order=np.asarray(["a", "b", "c"], dtype=object),
    )
    assert load_unified_artifact(path)["X"].shape == (2, 3)


def test_neighbour_recall_is_one_for_rigid_embedding() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(size=(30, 4))
    embedding = np.column_stack((values, np.zeros(len(values))))
    assert neighbour_recall(values, embedding, 5) == 1.0
    quality = embedding_quality(values, embedding, neighbours=(5, 20))
    assert quality["trustworthiness_5"] == 1.0
    assert "trustworthiness_20" not in quality


def test_predictive_stability_and_medoids_on_separated_groups() -> None:
    rng = np.random.default_rng(4)
    values = np.vstack(
        (rng.normal(-5, 0.1, size=(20, 2)), rng.normal(5, 0.1, size=(20, 2)))
    )
    stability, predictions = predictive_subsample_stability(
        values,
        lambda seed: KMeans(n_clusters=2, n_init=10, random_state=seed),
        seeds=(1, 2, 3),
    )
    assert stability == 1.0
    medoids = cluster_medoids(values, predictions[0])
    assert set(medoids) == {0, 1}
    for label, index in medoids.items():
        assert predictions[0][index] == label


def test_pca_caps_components() -> None:
    values = np.arange(60, dtype=float).reshape(10, 6)
    _, scores = fit_atlas_pca(values, n_components=100)
    assert scores.shape == (10, 6)


def test_density_stability_on_separated_groups() -> None:
    rng = np.random.default_rng(8)
    values = np.vstack(
        (rng.normal(-6, 0.05, size=(50, 2)), rng.normal(6, 0.05, size=(50, 2)))
    )
    from sklearn.cluster import HDBSCAN

    score = density_subsample_stability(
        values,
        lambda: HDBSCAN(min_cluster_size=10, min_samples=3),
        seeds=(1, 2, 3),
    )
    assert score == 1.0
