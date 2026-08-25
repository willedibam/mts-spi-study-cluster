import numpy as np

from src.process_features import (
    _edge_vectors,
    _pearson_corr_matrix,
    _parse_metrics,
    _rankdata,
    _spearman_corr_matrix,
)


def test_default_metric_is_pearson() -> None:
    assert _parse_metrics(None) == ["pearson"]


def test_rankdata_uses_average_ranks_for_ties() -> None:
    ranked = _rankdata(np.array([[1.0, 1.0, 3.0, 2.0]]))
    np.testing.assert_allclose(ranked, [[1.5, 1.5, 4.0, 3.0]])


def test_spearman_matrix_handles_ties_and_invalid_rows() -> None:
    values = np.array(
        [
            [1.0, 1.0, 2.0, 3.0],
            [4.0, 4.0, 2.0, 1.0],
            [1.0, np.nan, 2.0, 3.0],
        ]
    )
    result = _spearman_corr_matrix(values)
    assert np.isclose(result[0, 1], -1.0)
    assert np.isnan(result[0, 2])


def test_pearson_matrix_marks_constant_rows_invalid() -> None:
    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 3.0, 3.0, 3.0],
        ]
    )
    result = _pearson_corr_matrix(values)
    assert np.isclose(result[0, 0], 1.0)
    assert np.isnan(result[0, 1])
    assert np.isnan(result[1, 1])


def test_split_directed_vectors_share_dyad_order() -> None:
    matrix = np.arange(16, dtype=float).reshape(4, 4)
    entries = _edge_vectors("directed", matrix, directed=True, split_directed=True)
    upper = np.triu_indices(4, k=1)
    assert entries[0][0] == "directed__ij"
    assert entries[1][0] == "directed__ji"
    np.testing.assert_array_equal(entries[0][1], matrix[upper])
    np.testing.assert_array_equal(entries[1][1], matrix.T[upper])
