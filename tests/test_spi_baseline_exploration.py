import numpy as np
import pandas as pd

from scripts.spi_baseline_exploration import (
    summarize, var_population, project_features, retrieval,
)


def test_directed_mean_and_symmetrized_mean_agree_but_distributions_need_not():
    a = np.array([[np.nan, 1., 2.], [5., np.nan, 3.], [9., 7., np.nan]])
    ordered = summarize({"a": a}, ["a"])
    symmetric = summarize({"a": a}, ["a"], symmetric=True)
    assert ordered[0, 0] == symmetric[0, 0]
    assert ordered[0, 1] != symmetric[0, 1]
    assert ordered[0, 0] == 4.5  # diagonal never participates


def test_constant_validity_and_nonfinite_profiles_are_distinguished():
    constant = np.full((3, 3), 2.)
    np.fill_diagonal(constant, np.nan)
    bad = constant.copy()
    bad[0, 1] = np.nan
    values = summarize({"constant": constant, "bad": bad}, ["constant", "bad"])
    np.testing.assert_allclose(values[0], [2, 0, 2, 2, 2, 2, 2])
    assert np.isnan(values[1]).all()


def test_var_example_is_realizable_and_has_identical_marginals():
    for strength in (.02, .10):
        left = var_population(strength, -1, 3)
        right = var_population(strength, 1, 3)
        np.testing.assert_allclose(left[0], right[0])
        mask = ~np.eye(6, dtype=bool)
        np.testing.assert_allclose(np.sort(left[1][mask]), np.sort(right[1][mask]), atol=1e-15)
        for orientation, (c, lag, a, noise) in zip((-1, 1), (left, right)):
            assert np.linalg.eigvalsh(noise).min() > 0
            np.testing.assert_allclose(a @ c, lag, atol=1e-14)
            np.testing.assert_allclose(a @ c @ a.T + noise, c, atol=1e-14)
            np.testing.assert_allclose(np.corrcoef(c[mask], lag[mask])[0, 1], orientation)
            assert np.max(abs(np.linalg.eigvals(a))) < 1


def test_held_values_do_not_change_feature_selection_or_training_projection():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(40, 8))
    train[:, 7] = np.nan
    model1, reference1, _ = project_features(train, np.ones((2, 8)))
    model2, reference2, _ = project_features(train, np.full((2, 8), 1e9))
    assert 7 not in model1.transform.keep_indices
    np.testing.assert_array_equal(model1.transform.keep_indices, model2.transform.keep_indices)
    np.testing.assert_allclose(reference1, reference2)


def test_hard_retrieval_excludes_either_matching_observation_dimension():
    train = np.array([[0.], [.1], [.2], [1.], [2.]])
    query = np.array([[0.]])
    metadata = pd.DataFrame(dict(M=[8, 16, 8, 16, 16], T=[500, 500, 1000, 1000, 1000],
                                 label=["wrong", "wrong", "wrong", "right", "wrong"]))
    qmeta = pd.DataFrame(dict(M=[8], T=[500], label=["right"]))
    ap, top1 = retrieval(train, query, metadata, qmeta)
    np.testing.assert_allclose(ap, [1.])
    np.testing.assert_allclose(top1, [1.])
