import numpy as np

from src.representation_attribution import QUANTILES, rich_marginals, select_head


def test_rich_marginals_preserve_histograms_and_standardized_moments():
    a = np.arange(16, dtype=float).reshape(4, 4)**2
    mask = ~np.eye(4, dtype=bool)
    b = a.copy(); b[mask] = a[mask][::-1]
    np.testing.assert_allclose(rich_marginals({"a": a}, ["a"]), rich_marginals({"a": b}, ["a"]))
    x = rich_marginals({"a": a}, ["a"])
    y = rich_marginals({"a": 5 * a + 3}, ["a"])
    np.testing.assert_allclose(x[2:4], y[2:4])
    np.testing.assert_allclose(x[4:], np.quantile(a[mask], QUANTILES))
    unit = (a[mask] - a[mask].mean()) / a[mask].std()
    np.testing.assert_allclose(x[2:4], [np.mean(unit**3), np.mean(unit**4)])


def test_constant_and_nonfinite_marginal_semantics():
    a = np.ones((3, 3)); b = a.copy(); b[0, 1] = np.nan
    x = rich_marginals({"a": a, "b": b}, ["a", "b"]).reshape(2, 23)
    assert np.isnan(x[0, 2:4]).all() and np.isnan(x[1]).all()
    np.testing.assert_array_equal(x[0, :2], [1, 0])


def test_nonlinear_selection_does_not_use_held_out_rows():
    rng = np.random.default_rng(1)
    bank = {"m": rng.normal(size=(20, 8))}
    labels = np.tile([0, 1], 10)
    train = np.arange(12)
    pre = {"minimum_valid_fraction": .95, "variance_threshold": 1e-8,
           "z_scaling": "center", "pca_dimensions": 4, "pca_solver": "randomized", "pca_random_state": 1729}
    config = {"C_grid": [.1, 1, 10], "inner_folds": 2, "tolerance": .0001}
    left = select_head(bank, "m", train, labels, pre, config, 11, "rbf")
    bank["m"][12:] = np.nan; labels[12:] = 999
    right = select_head(bank, "m", train, labels, pre, config, 11, "rbf")
    assert left == right
