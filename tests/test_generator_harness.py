"""The runner learns everything generator-specific from the harness registry."""
from types import SimpleNamespace

import numpy as np

from src.generators import harness


def _spec(generator, **params):
    return SimpleNamespace(generator=generator, M=4, T=64, rng_seed=5, generator_params=params)


def test_plain_generator_yields_only_its_resolved_parameters():
    spec = _spec("gaussian_noise", zscore=True)
    data, extras = harness.generate_from_spec(spec)
    assert data.shape == (64, 4)
    assert set(extras) == {"resolved_params"} and extras["resolved_params"]["zscore"] is True
    assert harness.expected_files(spec) == [] and harness.semantics(spec, extras) == {}
    again, _ = harness.generate_from_spec(spec)
    assert np.array_equal(data, again)


def test_cml_full_lattice_is_requested_by_parameter():
    assert harness.expected_files(_spec("cml_logistic")) == []
    spec = _spec("cml_logistic", return_full_lattice=True, zscore=False)
    data, extras = harness.generate_from_spec(spec)
    assert harness.expected_files(spec) == ["full_lattice.npy"]
    assert data.shape == (64, 4) and np.asarray(extras["_full_lattice"]).shape[0] == 64


def test_every_ground_truth_generator_is_registered_consistently():
    from src.generators import GENERATOR_REGISTRY

    assert harness._GROUND_TRUTH and harness._GROUND_TRUTH <= set(harness._OUTPUTS)
    assert set(harness._SEMANTICS) == harness._GROUND_TRUTH
    assert set(harness._OUTPUTS) <= set(GENERATOR_REGISTRY)
    for name in harness._GROUND_TRUTH:
        assert harness.expected_files(_spec(name)) == ["ground_truth.npz"]
