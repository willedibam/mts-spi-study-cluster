import json

import numpy as np
import pytest

from research.representation.comparator_fill import blocks as B
from research.representation.comparator_fill import cohort, transfer


def synthetic(n_groups=12, per=8, T=200, M=4, K=5, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * per
    groups = np.repeat([f'p{k:02d}' for k in range(n_groups)], per)
    y = np.repeat(np.arange(n_groups) % 2, per)
    x = rng.normal(size=(n, T, M))
    x[:, :, 1] += (.2 + .8 * y[:, None]) * x[:, :, 0]
    edges = M * (M - 1)
    aligned = rng.normal(size=(n, K, edges)).astype(np.float32)
    aligned[:, 2] += y[:, None]
    aligned[::7, 4, 0] = np.nan
    pearson = np.array([np.corrcoef(w.T)[np.triu_indices(M, 1)] for w in x])
    return dict(x=x, y=y, group=groups, aligned=aligned.reshape(n, -1), pearson=pearson,
                z=rng.normal(size=(n, 20)) + .5 * y[:, None], m=rng.normal(size=(n, K * 23)),
                spectral=B.log_spectra(x), row_id=np.array([f'r{k}' for k in range(n)]),
                spi_order=np.array([f's{k}' for k in range(K)]))


def test_blocks_shapes_and_tangent_centre():
    d = synthetic()
    n, T, M = d['x'].shape
    assert B.mean_spi(d['aligned'], 5).shape == (n, 5)
    assert np.allclose(B.mean_spi(d['aligned'], 5)[0, 4], np.nanmean(d['aligned'].reshape(n, 5, -1)[0, 4]))
    assert B.catch22(d['x'][:3]).shape == (3, 22 * M)
    train = np.arange(0, n, 2)
    t = B.tangent(d['x'], train)
    assert t.shape == (n, M * (M + 1) // 2)
    assert np.abs(t[train].mean(0)).max() < 1e-6  # training windows are centred at their geometric mean


def test_log_spectra_matches_discovery_univariate():
    from research.representation.dataset_screen.prepare import univariate
    x = synthetic()['x'][:4]
    M = x.shape[2]
    assert np.allclose(B.log_spectra(x), np.array([univariate(w)[11 * M:43 * M] for w in x]))


def test_cohort_fit_predict_matches_original():
    from research.representation.character_axes import evaluate_cohorts
    d = synthetic()
    X = dict(pearson=d['pearson'], z=d['z'])
    train, test = np.arange(72), np.arange(72, 96)
    for names in (['pearson'], ['pearson', 'z']):
        new, _ = cohort.fit_predict(X, d['y'], d['group'], names, train, test)
        assert np.allclose(new, evaluate_cohorts.fit_predict(X, d['y'], d['group'], names, train, test))


def test_cohort_end_to_end(tmp_path):
    d = synthetic()
    np.savez(tmp_path / 'bank.npz', **d)
    out = tmp_path / 'out.json'
    cohort.CONTRASTS['toy'] = [('one_vs_zero', 1, 0)]
    cohort.main('toy', tmp_path, out, repeats=1, jobs=1, only=None)
    entry = json.loads(out.read_text())['contrasts']['one_vs_zero']
    assert {'pearson_z', 'spectral_z', 'mean_spi', 'm', 'aligned', 'single_spi', 'tangent', 'catch22', 'z_pc1', 'z_full'} <= set(entry['scores'])
    assert 'power' not in entry['scores']
    assert entry['single_spi_selected'].count('s2') >= 4  # the informative SPI is found in nearly every fold
    assert entry['scores']['tangent']['auroc'] > .9 and len(entry['unit_keys']) == 12
    assert 'pearson_z - pearson' in entry['contrasts']


def test_transfer_wrapper_matches_statistical_fit():
    from research.representation.dataset_screen import statistical
    d = synthetic(n_groups=18)
    n, M = len(d['y']), d['x'].shape[2]
    bank = {k: d[k] for k in ('m', 'z', 'pearson', 'aligned', 'y', 'group', 'row_id', 'spi_order')}
    bank.update(validity=np.ones((n, 3)), u=np.zeros((n, 75 * M)), role=np.where(np.arange(n) < 96, 'source', 'screen'))
    bank['u'][:, 11 * M:43 * M] = B.log_spectra(d['x'])
    source_ix, screen_ix = np.flatnonzero(bank['role'] == 'source'), np.flatnonzero(bank['role'] == 'screen')
    bank = transfer.extend(bank, d['x'], source_ix)
    source = statistical.source_rows(bank)
    settings = dict(pca_caps=[2, 8], ridge_alphas=[.1, 10])
    fitted, _ = transfer.fit(source, 'pearson_z', settings)
    reference, _ = statistical.fit_one(source, 'pearson', settings, add_z=True)
    assert np.allclose(transfer.predict(fitted, bank, 'pearson_z', screen_ix), statistical.predict(reference, bank, screen_ix))
    predictions = {}
    for name in ('spectra_z', 'mean_spi', 'tangent', 'z_pc1', 'z_full'):
        fitted, report = transfer.fit(source, name, settings)
        predictions[name] = transfer.predict(fitted, bank, name, screen_ix)
        assert predictions[name].shape == (len(screen_ix), 2)
    scored = transfer.score(bank['y'][screen_ix], bank['group'][screen_ix], predictions, dict(z=predictions['z_full']))
    assert scored['methods']['tangent']['auroc_pooled'] > .9
    assert 'z_pc1 versus z' in scored['contrasts']
