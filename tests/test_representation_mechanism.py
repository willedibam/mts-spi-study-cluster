import numpy as np

from src.representation_mechanism import permute_dyads, linear_dynamics_features
from src.spi_spi_contract import build_unified_feature_values
from src.mpi_representation_baselines import summarize_mpis


def test_independent_dyad_maps_preserve_marginals_reciprocity_and_validity():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(6, 6))
    mpis = {"a": a, "b": a**2, "sym": a + a.T, "bad": a * np.nan}
    order = list(mpis)
    moved = permute_dyads(mpis, order, rng)
    original = summarize_mpis(mpis, order)
    perturbed = summarize_mpis(moved, order)
    np.testing.assert_allclose(original[0], perturbed[0], equal_nan=True)
    np.testing.assert_array_equal(original[2], perturbed[2])
    np.testing.assert_allclose(original[1].reshape(4, 9)[:, 6], perturbed[1].reshape(4, 9)[:, 6], equal_nan=True)
    np.testing.assert_array_equal(moved["sym"], moved["sym"].T)
    mask = ~np.eye(6, dtype=bool)
    for name in order:
        np.testing.assert_array_equal(np.sort(mpis[name][mask]), np.sort(moved[name][mask]))
    z = build_unified_feature_values(mpis, order)[0]
    null = build_unified_feature_values(moved, order)[0]
    np.testing.assert_array_equal(np.isfinite(z), np.isfinite(null))
    assert not np.allclose(z, null, equal_nan=True)


def test_shared_dyad_map_preserves_all_spi_correlations():
    rng = np.random.default_rng(17)
    mpis = {str(i): rng.normal(size=(8, 8)) for i in range(5)}
    order = list(mpis)
    moved = permute_dyads(mpis, order, rng, shared=True)
    np.testing.assert_allclose(build_unified_feature_values(mpis, order)[0],
                               build_unified_feature_values(moved, order)[0], atol=1e-14)


def test_direct_linear_features_recover_known_dynamics_and_ignore_channel_names():
    rng = np.random.default_rng(9)
    m = 5
    transition = .35 * np.eye(m) + .12 * np.roll(np.eye(m), 1, axis=1)
    raw = rng.normal(size=(14000, m))
    for t in range(1, len(raw)):
        raw[t] += transition @ raw[t - 1]
    values = linear_dynamics_features(raw)
    np.testing.assert_allclose(values, [.35, .12], atol=.015)
    np.testing.assert_allclose(values, linear_dynamics_features(3 * raw[:, [3, 0, 4, 1, 2]] + 7), atol=1e-12)
