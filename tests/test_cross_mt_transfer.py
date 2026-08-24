import numpy as np

from src.cross_mt_transfer import (
    geometry_metrics,
    pooled_baseline_features,
    retrieval_scores,
    stratified_bootstrap_interval,
)


def test_pooled_baseline_dimensions_do_not_depend_on_M_or_T() -> None:
    rng = np.random.default_rng(4)
    small = pooled_baseline_features(rng.normal(size=(500, 8)))
    large = pooled_baseline_features(rng.normal(size=(1000, 16)))

    assert small.keys() == large.keys()
    for name in small:
        assert small[name][0].shape == large[name][0].shape
        assert small[name][1] == large[name][1]
    assert small["pooled_univariate"][0].shape == (50,)
    assert small["pooled_dependence"][0].shape == (32,)
    assert small["pooled_combined"][0].shape == (82,)


def test_cross_size_retrieval_scores_known_neighbours() -> None:
    query = np.asarray([[0.0], [10.0]])
    gallery = np.asarray([[0.1], [0.2], [9.8], [9.9]])
    scores = retrieval_scores(
        query,
        np.asarray(["a", "b"]),
        gallery,
        np.asarray(["a", "a", "b", "b"]),
    )

    np.testing.assert_array_equal(scores["recall_at_1"], [1.0, 1.0])
    np.testing.assert_array_equal(scores["recall_at_5"], [1.0, 1.0])
    np.testing.assert_allclose(scores["average_precision"], [1.0, 1.0])


def test_geometry_distinguishes_class_from_cell_effect() -> None:
    labels = np.repeat(["a", "b"], 4)
    cells = np.tile(np.repeat(["small", "large"], 2), 2)
    coordinates = np.asarray(
        [[0.0], [0.1], [0.2], [0.3], [10.0], [10.1], [10.2], [10.3]]
    )

    result = geometry_metrics(coordinates, labels, cells)

    assert result["class_eta_squared"] > 0.99
    assert result["cell_eta_squared"] < 0.001
    assert result["class_to_cell_ratio"] > 1000
    assert result["matched_distance_ratio"] < 0.1


def test_stratified_bootstrap_is_deterministic() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 3.0])
    strata = np.asarray(["a", "a", "b", "b"])
    metric = lambda index: float(np.mean(values[index]))

    first = stratified_bootstrap_interval(
        metric, strata, repetitions=50, confidence_level=0.9, random_state=7
    )
    second = stratified_bootstrap_interval(
        metric, strata, repetitions=50, confidence_level=0.9, random_state=7
    )

    assert first == second
