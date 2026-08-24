import numpy as np
import pytest

from src.process_features import (
    _edge_vectors,
    _mutual_information_matrix,
    _pearson_corr_matrix,
    _spearman_corr_matrix,
)
from src.spi_spi_contract import build_feature_blocks


def _schema_index(result, relation: str, first: str, second: str) -> int:
    for index, feature in enumerate(result.dir_schema):
        if (feature.relation, feature.spi_a, feature.spi_b) == (
            relation,
            first,
            second,
        ):
            return index
    raise AssertionError(f"feature not found: {(relation, first, second)}")


def _example_mpis() -> dict[str, np.ndarray]:
    return {
        "u": np.array(
            [
                [9.0, 1.0, 4.0, 2.0],
                [1.0, 8.0, 3.0, 7.0],
                [4.0, 3.0, 6.0, 5.0],
                [2.0, 7.0, 5.0, 4.0],
            ]
        ),
        "d1": np.array(
            [
                [1.0, 2.0, 8.0, 5.0],
                [7.0, 2.0, 4.0, 9.0],
                [3.0, 6.0, 3.0, 1.0],
                [4.0, 0.0, 7.0, 4.0],
            ]
        ),
        "d2": np.array(
            [
                [4.0, 8.0, 1.0, 6.0],
                [2.0, 3.0, 7.0, 5.0],
                [9.0, 0.0, 2.0, 4.0],
                [3.0, 1.0, 8.0, 1.0],
            ]
        ),
    }


def test_common_channel_permutation_leaves_both_blocks_unchanged() -> None:
    mpis = _example_mpis()
    order = ["u", "d1", "d2"]
    directed = [False, True, True]
    original = build_feature_blocks(mpis, order, directed)

    permutation = np.array([2, 0, 3, 1])
    permuted = {
        name: matrix[np.ix_(permutation, permutation)]
        for name, matrix in mpis.items()
    }
    transformed = build_feature_blocks(permuted, order, directed)

    np.testing.assert_allclose(transformed.z_sym, original.z_sym, atol=1e-7)
    np.testing.assert_allclose(transformed.z_dir, original.z_dir, atol=1e-7)
    assert transformed.sym_schema == original.sym_schema
    assert transformed.dir_schema == original.dir_schema


def test_transpose_one_directed_spi_swaps_parallel_and_reverse() -> None:
    mpis = _example_mpis()
    order = ["u", "d1", "d2"]
    directed = [False, True, True]
    original = build_feature_blocks(mpis, order, directed)
    transposed = build_feature_blocks(
        {**mpis, "d1": mpis["d1"].T}, order, directed
    )

    parallel = _schema_index(original, "parallel", "d1", "d2")
    reverse = _schema_index(original, "reverse", "d1", "d2")
    np.testing.assert_allclose(
        transposed.z_dir[parallel], original.z_dir[reverse], atol=1e-7
    )
    np.testing.assert_allclose(
        transposed.z_dir[reverse], original.z_dir[parallel], atol=1e-7
    )
    reciprocity = _schema_index(original, "reciprocity", "d1", "d1")
    np.testing.assert_allclose(
        transposed.z_dir[reciprocity], original.z_dir[reciprocity], atol=1e-7
    )


def test_transpose_every_directed_spi_preserves_feature_vector() -> None:
    mpis = _example_mpis()
    order = ["u", "d1", "d2"]
    directed = [False, True, True]
    original = build_feature_blocks(mpis, order, directed)
    transposed = build_feature_blocks(
        {
            name: matrix.T if is_directed else matrix
            for name, matrix, is_directed in zip(
                order, (mpis[name] for name in order), directed
            )
        },
        order,
        directed,
    )
    np.testing.assert_allclose(transposed.z_sym, original.z_sym, atol=1e-7)
    np.testing.assert_allclose(transposed.z_dir, original.z_dir, atol=1e-7)


def test_antisymmetric_information_survives_directional_block() -> None:
    first = np.array(
        [[0.0, 1.0, 2.0], [-1.0, 0.0, 4.0], [-2.0, -4.0, 0.0]]
    )
    second = np.array(
        [[0.0, 3.0, -1.0], [-3.0, 0.0, 2.0], [1.0, -2.0, 0.0]]
    )
    result = build_feature_blocks(
        {"a": first, "b": second}, ["a", "b"], [True, True]
    )

    assert np.isnan(result.z_sym[0])
    assert np.isfinite(result.z_dir).all()
    for name in ("a", "b"):
        index = _schema_index(result, "reciprocity", name, name)
        np.testing.assert_allclose(result.z_dir[index], -1.0, atol=1e-7)


@pytest.mark.parametrize("metric", ["pearson", "spearman", "mi"])
def test_sym_block_exactly_matches_legacy_nan_contract(metric: str) -> None:
    mpis = _example_mpis()
    order = ["u", "d1", "d2"]
    directed = [False, True, True]
    result = build_feature_blocks(mpis, order, directed, metric=metric)

    legacy_vectors = np.vstack(
        [
            _edge_vectors(name, mpis[name], is_directed, split_directed=False)[0][1]
            for name, is_directed in zip(order, directed)
        ]
    )
    correlation_function = {
        "pearson": _pearson_corr_matrix,
        "spearman": _spearman_corr_matrix,
        "mi": _mutual_information_matrix,
    }[metric]
    legacy = correlation_function(legacy_vectors)[np.triu_indices(3, k=1)]
    np.testing.assert_array_equal(result.z_sym, legacy.astype(np.float32))


def test_all_undirected_mpis_add_no_duplicate_directional_features() -> None:
    mpis = _example_mpis()
    result = build_feature_blocks(mpis, list(mpis), [False, False, False])
    assert result.z_dir.size == 0
    assert result.dir_schema == ()


def test_diagonals_are_excluded_from_every_block() -> None:
    mpis = _example_mpis()
    order = ["u", "d1", "d2"]
    directed = [False, True, True]
    original = build_feature_blocks(mpis, order, directed)
    changed = {name: matrix.copy() for name, matrix in mpis.items()}
    for offset, matrix in enumerate(changed.values(), start=1):
        np.fill_diagonal(matrix, 1e9 * offset)
    transformed = build_feature_blocks(changed, order, directed)
    np.testing.assert_array_equal(transformed.z_sym, original.z_sym)
    np.testing.assert_array_equal(transformed.z_dir, original.z_dir)


def test_undefined_correlations_remain_nan_with_reasons() -> None:
    constant = np.ones((3, 3), dtype=float)
    varying = np.array(
        [[0.0, 1.0, 2.0], [4.0, 0.0, 3.0], [5.0, 6.0, 0.0]]
    )
    result = build_feature_blocks(
        {"constant": constant, "varying": varying},
        ["constant", "varying"],
        [True, True],
    )
    assert np.isnan(result.z_sym[0])
    assert not result.sym_valid[0]
    assert np.isnan(result.z_dir).any()
    assert result.invalid_reasons["constant"] == {
        "sym": "constant",
        "ordered": "constant",
    }
