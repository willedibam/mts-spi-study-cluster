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
    return yaml.safe_load(Path("configs/analysis/representation-stage-b-260907.yaml").read_text())


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


def test_end_to_end_state_pilot_groups_views_and_rejects_changed_data_protocol(tmp_path):
    import json
    from scripts.build_representation_state_data import build
    from scripts.run_representation_state_pilot import run
    from scripts.report_representation_state_pilot import report
    from scripts.run_representation_state_random_control import run as run_random
    from scripts.run_representation_state_raw_control import run as run_raw
    from src.representation_state_data import load_state_data
    p = protocol()
    p.pop("data_protocol")
    p["generator"].update(M=4, N_full=4, T=64, future_truth_T=64, burn_time=1., reduced_couplings=[.6, 1.4])
    p["observations"].update(source={"M": 4, "T": 64}, M_values=[2, 4], T_values=[32, 64])
    p["sampling"].update(training_masters_per_coupling=4, evaluation_masters_per_coupling=2,
                         labelled_training_masters_per_coupling=[2], total_label_budgets=[4])
    p["methods"]["subset_seeds"] = [11]
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(p))
    data = tmp_path / "data"
    build(config, data)
    run(config, data, tmp_path / "simple", ["mean", "observables"], "cpu")
    run_random(config, data, tmp_path / "random", "cpu")
    run_raw(config, data, tmp_path / "raw")
    report(config, data, [tmp_path / "simple", tmp_path / "random", tmp_path / "raw"], tmp_path / "report")
    result = json.loads((tmp_path / "report/results.json").read_text())
    assert result["evaluation_masters"] == 4
    assert len(result["summary"]["both_M_and_T_changed"]["mean"]["MAE"]) == 1
    assert len(result["summary"]["both_M_and_T_changed"]["random_encoder"]["MAE"]) == 1
    with np.load(tmp_path / "raw/features.npz") as features:
        assert features["pooled_raw"].shape == (24, 82)
        assert features["pooled_raw_phase"].shape == (24, 83)
        np.testing.assert_array_equal(features["pooled_raw_phase"][:, -1], np.load(data / "observables.npy")[:, 1])
    assert "pooled_raw_phase_minus_pooled_raw" in result["paired_primary_comparisons"]
    assert "M2_T32" in result["summary"]
    with np.load(tmp_path / "random/initial-features-s11.npz") as features:
        assert features["X"].shape == (24, 128)
    fit = json.loads((tmp_path / "simple/observables-n2-s11.json").read_text())
    assert fit["labels_total"] == 4
    assert set(fit["train_indices"]).isdisjoint(fit["evaluation_indices"])
    # A fitting-only revision can reuse raw masters, a changed observation
    # protocol cannot silently inherit their provenance.
    p["data_protocol"] = str(config)
    p["methods"]["raw_encoder_spec"]["dropout"] = .1
    revision = tmp_path / "revision.yaml"
    revision.write_text(yaml.safe_dump(p))
    load_state_data(data, revision)
    p["observations"]["source"]["M"] = 2
    revision.write_text(yaml.safe_dump(p))
    with pytest.raises(ValueError, match="data construction"):
        load_state_data(data, revision)
