import json

import numpy as np
import pytest

from research.representation.dataset_screen.common import folds, weights, probabilities, metrics
from research.representation.dataset_screen.statistical import fit_one, fit_projection, predict, feature_bank
from research.representation.dataset_screen.raw import normalization, load_role, neural_spec, make_network, fit_network
from research.representation.shared.interaction_share_learning import ridge_readout


def small_bank():
    rng = np.random.default_rng(12)
    y = np.tile(np.arange(3), 9)
    return dict(m=rng.normal(size=(27, 8)) + y[:, None], z=rng.normal(size=(27, 15)),
                y=y, group=np.repeat([f'p{i}' for i in range(9)], 3), row_id=np.array([f'r{i}' for i in range(27)]))


def test_grouped_multiclass_folds_and_metrics_ignore_duplicate_windows():
    bank = small_bank()
    y, g = bank['y'], bank['group']
    seen = []
    for train, valid in folds(y, g):
        assert set(g[train]).isdisjoint(g[valid])
        assert set(y[train]) == set(y[valid]) == {0, 1, 2}
        seen.extend(valid)
    assert sorted(seen) == list(range(27))
    p = probabilities(np.random.default_rng(7).uniform(size=(27, 3)))
    repeated = np.r_[np.tile(np.arange(3), 20), np.arange(3, 27)]
    a, b = metrics(y, p, g), metrics(y[repeated], p[repeated], g[repeated])
    for key in a: assert a[key] == pytest.approx(b[key])
    assert [weights(y, g)[y == c].sum() for c in range(3)] == pytest.approx([9, 9, 9])
    np.testing.assert_allclose(probabilities([[-1, -1, -1]]), [[1/3]*3])


@pytest.mark.parametrize('method', ['m', 'm_rbf'])
def test_multiclass_source_fit_and_fusion_null_preserve_baseline(method):
    bank = small_bank(); ix = np.arange(27); target = np.eye(3)[bank['y']]
    transform, x = fit_projection(bank, 'm', ix, 3)
    fused, fx = fit_projection(bank, 'm', ix, 3, True, 0.)
    np.testing.assert_allclose(x, fx[:, :3])
    np.testing.assert_array_equal(fx[:, 3:], 0)
    head = 'rbf' if method.endswith('_rbf') else 'pca'
    a = ridge_readout(x, target, head, 1)
    b = ridge_readout(fx, target, head, 1)
    np.testing.assert_allclose(a.predict(x), b.predict(fx), atol=1e-12)
    fit, report = fit_one(bank, method, dict(pca_caps=[3], ridge_alphas=[1]), add_z=True)
    assert np.isfinite(report['source_validation_brier'])
    assert predict(fit, bank, ix).shape == (27, 3)
    assert len(report['candidates']) == 3


def test_projection_is_not_affected_by_heldout_extremes():
    bank = small_bank(); train = np.arange(18)
    first, x = fit_projection(bank, 'm', train, 3, True)
    altered = {k:v.copy() for k,v in bank.items()}
    altered['m'][18:] = 1e10; altered['z'][18:] = np.nan
    second, y = fit_projection(altered, 'm', train, 3, True)
    np.testing.assert_allclose(x, y)
    np.testing.assert_allclose(first.transform(bank, train), second.transform(altered, train))


def test_aligned_catalogue_control_keeps_spi_and_channel_identity():
    bank = small_bank()
    bank['aligned'] = np.random.default_rng(3).normal(size=(27, 48))
    all_spis, view = feature_bank(bank, 'aligned')
    assert view == 'u'
    np.testing.assert_array_equal(all_spis['u'], bank['aligned'])
    one, _ = feature_bank(bank, 'single_spi', 2, 4)
    np.testing.assert_array_equal(one['u'], bank['aligned'][:, 24:36])


def test_raw_source_loader_and_scaler_do_not_read_heldout_data(tmp_path):
    x = np.random.default_rng(2).normal(size=(2, 32, 3)).astype('float32')
    np.savez(tmp_path / 'raw-views.npz', a=x[0], b=x[1])
    rows = [dict(row_id=name, group=name, target=i, role='source') for i,name in enumerate(['a','b'])]
    rows.append(dict(row_id='absent', group='heldout', target=0, role='confirmation'))
    found, y, groups, _ = load_role(tmp_path, dict(rows=rows), 'source')
    np.testing.assert_array_equal(found, x)
    mean, scale = normalization(found)
    np.testing.assert_allclose(mean, x.mean((0, 1)), atol=1e-7)
    np.testing.assert_allclose(((x-mean)/scale).std((0, 1)), 1, atol=1e-6)


@pytest.mark.parametrize('method', ['inceptiontime', 'aligned_encoder'])
def test_network_multiclass_training_and_legacy_output(method):
    import torch
    torch.set_num_threads(1)
    spec, _, _ = neural_spec(method, 4, 17)
    x = torch.randn(3, 96, 4)
    model = make_network(method, spec).eval()
    with torch.no_grad():
        assert model(x).shape == (3, 17)
    binary = dict(spec, output_classes=2 if method == 'inceptiontime' else 1)
    binary.pop('return_binary_difference', None)
    legacy = make_network(method, binary).eval()
    with torch.no_grad():
        old = legacy(x)
        assert old.shape == (3,)
        if method == 'inceptiontime':
            legacy.binary_difference = False
            logits = legacy(x)
            torch.testing.assert_close(old, logits[:, 1] - logits[:, 0])
    fitted, report = fit_network(method, x, np.arange(3), np.array(['a','a','a']),
                                {**spec, 'output_classes':3}, {}, .001, 11, epochs=1)
    assert np.isfinite(report['source_training_metrics']['brier'])
