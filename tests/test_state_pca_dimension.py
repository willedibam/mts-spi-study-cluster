import numpy as np

from scripts.check_state_pca_dimension import select_cap


def test_pca_cap_and_ridge_selection_ignore_held_out_data_and_labels():
    rng = np.random.default_rng(42)
    bank = {"z": rng.normal(size=(30, 12))}
    train = np.arange(16)
    strata = np.tile(np.arange(4), 8)[:30]
    target = rng.uniform(size=30)
    protocol = {"methods": {"ridge_alpha_grid": [.1, 1], "preprocessing": {
        "minimum_valid_fraction": .95, "variance_threshold": 1e-8, "z_scaling": "center",
        "pca_dimensions": 32, "pca_solver": "randomized", "pca_random_state": 1729}}}
    a = select_cap(bank, "z", train, target, strata, protocol, [1, 2, 4, 8], 11)
    bank["z"][16:] = np.nan
    target[16:] = 999
    b = select_cap(bank, "z", train, target, strata, protocol, [1, 2, 4, 8], 11)
    assert a == b
