import numpy as np

from src.spi_spi_analysis import fit_feature_transform, fit_frozen_pc1


def test_transform_fits_missingness_and_variance_on_development_only() -> None:
    development = np.array(
        [
            [1.0, 5.0, np.nan, 0.0],
            [2.0, 5.0, 2.0, 2.0],
            [3.0, 5.0, 4.0, 4.0],
        ]
    )
    model = fit_feature_transform(
        development,
        ["sym", "sym", "dir", "dir"],
        minimum_valid_fraction=2 / 3,
        variance_threshold=0.1,
    )
    np.testing.assert_array_equal(model.keep_indices, [0, 2, 3])
    transformed = model.transform(np.array([[4.0, 100.0, np.nan, 6.0]]))
    assert transformed.shape == (1, 3)
    assert np.isfinite(transformed).all()
    # The held-out value cannot change the development median used for imputation.
    assert transformed[0, 1] == 0.0


def test_block_balancing_equalizes_total_development_variance() -> None:
    rng = np.random.default_rng(2)
    development = np.column_stack(
        (
            rng.normal(size=(200, 2)),
            rng.normal(scale=4.0, size=(200, 7)),
        )
    )
    model = fit_feature_transform(
        development,
        ["sym"] * 2 + ["dir"] * 7,
        block_balanced=True,
    )
    transformed = model.transform(development)
    sym_total = np.var(transformed[:, model.kept_blocks == "sym"], axis=0).sum()
    dir_total = np.var(transformed[:, model.kept_blocks == "dir"], axis=0).sum()
    np.testing.assert_allclose(sym_total, 1.0, atol=1e-12)
    np.testing.assert_allclose(dir_total, 1.0, atol=1e-12)


def test_pc1_sign_is_target_independent_and_deterministic() -> None:
    rng = np.random.default_rng(4)
    development = rng.normal(size=(80, 10))
    first = fit_frozen_pc1(development, ["sym"] * 10)
    second = fit_frozen_pc1(development, ["sym"] * 10)
    np.testing.assert_array_equal(first.component, second.component)
    anchor = np.argmax(np.abs(first.component))
    assert first.component[anchor] > 0
