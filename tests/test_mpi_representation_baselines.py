import numpy as np

from src.mpi_representation_baselines import summarize_mpis, pearson_geometry_audit
from src.spi_spi_contract import build_unified_feature_values


def test_node_relabeling_and_diagonals_do_not_change_summaries():
    rng = np.random.default_rng(8)
    mpis = {str(i): rng.normal(size=(5, 5)) for i in range(3)}
    permutation = [3, 1, 4, 0, 2]
    moved = {n: a[np.ix_(permutation, permutation)].copy() for n, a in mpis.items()}
    for a in moved.values():
        np.fill_diagonal(a, np.nan)
    for left, right in zip(summarize_mpis(mpis, list(mpis)), summarize_mpis(moved, list(mpis))):
        np.testing.assert_allclose(left, right, atol=1e-12)


def test_same_edge_histogram_can_have_different_graph_summary():
    star = np.zeros((4, 4)); path = np.zeros((4, 4))
    for a, b in [(0, 1), (0, 2), (0, 3)]:
        star[a, b] = star[b, a] = 1
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        path[a, b] = path[b, a] = 1
    a = summarize_mpis({"x": star}, ["x"])
    b = summarize_mpis({"x": path}, ["x"])
    np.testing.assert_allclose(a[0], b[0])
    assert not np.allclose(a[1], b[1])


def test_constant_is_valid_for_marginals_but_not_correlation():
    zero = np.zeros((3, 3)); bad = zero.copy(); bad[0, 1] = np.nan
    m, g, valid = summarize_mpis({"zero": zero, "bad": bad}, ["zero", "bad"])
    np.testing.assert_array_equal(m[:7], np.zeros(7))
    assert np.isnan(m[7:]).all() and np.isnan(g[6:]).all()
    assert not valid.any()


def test_shared_edge_permutation_preserves_z_and_marginals():
    rng = np.random.default_rng(2)
    mpis = {str(i): rng.normal(size=(4, 4)) for i in range(4)}
    mask = ~np.eye(4, dtype=bool)
    permutation = rng.permutation(mask.sum())
    moved = {name: a.copy() for name, a in mpis.items()}
    for name in moved:
        moved[name][mask] = mpis[name][mask][permutation]
    np.testing.assert_allclose(summarize_mpis(mpis, list(mpis))[0], summarize_mpis(moved, list(mpis))[0])
    np.testing.assert_allclose(build_unified_feature_values(mpis, list(mpis))[0],
                               build_unified_feature_values(moved, list(mpis))[0])
    assert not np.allclose(summarize_mpis(mpis, list(mpis))[1], summarize_mpis(moved, list(mpis))[1])


def test_correlation_geometry_rank_bound():
    rng = np.random.default_rng(12)
    mpis = {str(i): rng.normal(size=(3, 3)) for i in range(12)}
    result = pearson_geometry_audit(mpis, list(mpis))
    assert result["rank_upper_bound"] == 5
    assert result["numerical_rank"] == 5
    assert result["minimum_eigenvalue"] > -1e-12
