import copy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.representation_state_data import observed_view, simple_observables
from src.representation_state_neural import AlignedChannelEncoder, fit_encoder, seed_torch
from scripts.run_representation_state_pilot import select_ridge


def protocol():
    return yaml.safe_load(Path("configs/analysis/representation-stage-b-proposal-260907.yaml").read_text())


def test_views_keep_a_common_endpoint_and_channel_order():
    master = np.random.default_rng(2).normal(size=(1000, 32))
    np.testing.assert_array_equal(observed_view(master, 8, 500), observed_view(master, 16, 1000)[500:, :8])
    with pytest.raises(ValueError):
        observed_view(master, 33, 500)


def test_observables_use_observed_cosines_and_are_permutation_invariant():
    t = np.linspace(0, 16 * np.pi, 1024, endpoint=False)
    synchronized = np.tile(np.cos(t)[:, None], (1, 8))
    np.testing.assert_allclose(simple_observables(synchronized), [1, 1], atol=1e-12)
    asynchronous = np.cos(t[:, None] + np.linspace(0, 2 * np.pi, 8, endpoint=False))
    assert simple_observables(asynchronous)[1] < 1e-12
    np.testing.assert_allclose(simple_observables(asynchronous[:, ::-1]), simple_observables(asynchronous), atol=1e-12)


def test_encoder_permutation_variable_shapes_and_cross_channel_interaction():
    torch.set_num_threads(2)
    seed_torch(4)
    model = AlignedChannelEncoder(protocol()["methods"]["raw_encoder_spec"]).eval()
    with torch.no_grad():
        for m, t in [(8, 500), (16, 1000), (32, 500)]:
            x = torch.randn(1, t, m)
            a = model(x)
            b = model(x[:, :, torch.randperm(m)])
            assert a.shape == (1,) and torch.isfinite(a).all()
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        x = torch.randn(1, 128, 8)
        tokens = model.tokens(x)
        x[:, :, 1:] = 0
        changed = model.tokens(x)
        # Channel zero's local waveform is unchanged; its contextual tokens
        # must respond to other channels before the final pooling operation.
        assert (tokens[:, :, 0] - changed[:, :, 0]).abs().max() > 1e-4


def test_ridge_selection_cannot_access_test_values_or_targets():
    p = protocol()
    rng = np.random.default_rng(4)
    bank = {"m": rng.normal(size=(24, 10))}
    y = rng.uniform(size=24)
    strata = np.tile([0, 1], 12)
    train = np.arange(16)
    chosen, log = select_ridge(bank, "m", train, y, strata, p, 11)
    bank["m"][16:] = 1e9
    y[16:] = -1e9
    again, new_log = select_ridge(bank, "m", train, y, strata, p, 11)
    assert chosen == again and log == new_log
    for fold in log["folds"]:
        assert set(fold["fit"]).isdisjoint(fold["validation"])
        assert set(fold["fit"] + fold["validation"]) == set(train)


def test_final_neural_fit_requires_frozen_epoch_count_and_has_gradients():
    torch.set_num_threads(2)
    spec = copy.deepcopy(protocol()["methods"]["raw_encoder_spec"])
    x = torch.randn(4, 64, 4)
    y = torch.tensor([.1, .3, .6, .9])
    with pytest.raises(ValueError):
        fit_encoder(x, y, spec, .001, .0001, 7)
    fitted, log = fit_encoder(x, y, spec, .001, .0001, 7, epochs=2)
    assert log["epochs_run"] == 2 and log["validation_MAE"] is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fitted.parameters())
