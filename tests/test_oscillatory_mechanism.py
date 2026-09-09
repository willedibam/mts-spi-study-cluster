from pathlib import Path

import numpy as np
import yaml

from src.oscillatory_mechanism import draw_parameters, integrate_phase, simulate


def test_uncoupled_solution_and_group_mean_noise_conservation():
    rng = np.random.default_rng(1729)
    initial = rng.normal(size=32)
    brownian = rng.normal(size=(200, 32)) * np.sqrt(.0005)
    exact = initial + 2 * np.cumsum(brownian, axis=0)[19::20]
    for stride in [1, 2]:
        uncoupled = integrate_phase(initial, brownian, 0., 2., .0005, stride, 20)
        coupled = integrate_phase(initial, brownian, 60., 2., .0005, stride, 20)
        np.testing.assert_allclose(uncoupled, exact, atol=1e-13)
        # Antisymmetric coupling cancels in each unwrapped group mean.
        np.testing.assert_allclose(coupled.reshape(-1, 4, 8).mean(2),
                                   exact.reshape(-1, 4, 8).mean(2), atol=1e-13)


def test_conditions_and_resolution_keep_nuisance_and_phase_streams():
    settings = yaml.safe_load(Path("configs/analysis/oscillatory-mechanism-scout-260910.yaml").read_text())["generator"]
    settings = dict(settings, burn_seconds=.1)
    parameters = draw_parameters(999, settings)
    x, a = simulate(False, 778, parameters, settings, t=100, return_latent=True)
    y, b = simulate(True, 778, parameters, settings, t=100, return_latent=True)
    np.testing.assert_array_equal(a["phase"], b["phase"])
    assert not np.array_equal(x, y)
    np.testing.assert_array_equal(x, simulate(False, 778, parameters, settings, t=100))
    _, fine = simulate(False, 778, parameters, settings, t=100, dt=.0005, return_latent=True)
    np.testing.assert_array_equal(a["logamp"], fine["logamp"])
    # Different step sizes receive exactly the same group-average Brownian path.
    inverse = np.argsort(parameters["order"])
    np.testing.assert_allclose(a["phase"][:, inverse].reshape(-1, 4, 8).mean(2),
                               fine["phase"][:, inverse].reshape(-1, 4, 8).mean(2), atol=1e-12)
