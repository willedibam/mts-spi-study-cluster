"""Comparator ladder on the fMRI rest/film bank, under its frozen protocol.

Protocol (unchanged from empirical_transfer.fit_source and evaluate_fmri): source-only selection over the
configured PCA caps and ridge penalties on three participant folds, person/state-balanced weights, frozen fits
scored at threshold 0.5 as equally weighted participant means in each held-out role.

Added here: spectra + z, r + spectra (+ z), mean-SPI (+ z), tangent-space covariance, per-channel catch22, and
z read through its first PC or at full rank. The tangent reference is fitted on source rows. Windows in this
bank are standardized, so the tangent block carries correlation, not channel power.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from mtsspi.provenance import atomic_json, file_sha256
from research.representation.comparator_fill import blocks as B
from research.representation.empirical_transfer import fit_source
from research.representation.empirical_transfer.evaluate_fmri import ROLES, group_metrics

# name -> (view, PCA caps override)
METHODS = {'spectral-z': ('spectral+z', None), 'pearson-spectral': ('pearson_spectral', None),
           'pearson-spectral-z': ('pearson_spectral+z', None), 'mean-spi': ('mean_spi', None),
           'mean-spi-z': ('mean_spi+z', None), 'tangent': ('tangent', None), 'catch22': ('catch22', None),
           'z-pc1': ('z', [1]), 'z-full': ('z', [10 ** 6])}
PAIRS = [('spectral-z', 'spectral'), ('pearson-spectral-z', 'pearson-spectral'), ('mean-spi-z', 'mean-spi'),
         ('pearson-z', 'pearson'), ('z-pca', 'pearson'), ('z-pca', 'mean-spi'), ('z-pca', 'spectral'),
         ('z-pca', 'tangent'), ('z-pca', 'catch22'), ('z-full', 'z-pca'), ('z-pc1', 'z-pca')]


def load(root):
    if file_sha256(root / 'bank.npz') != json.loads((root / 'bank.json').read_text())['sha256']:
        raise ValueError('bank checksum mismatch')
    keys = ['x', 'z', 'aligned', 'pearson', 'spectral', 'row_id', 'group', 'role', 'y']
    with np.load(root / 'bank.npz', allow_pickle=False) as archive:
        bank = {k: archive[k] for k in keys}
    x = np.asarray(bank.pop('x'), dtype=np.float64)
    M = x.shape[2]
    source = np.flatnonzero(bank['role'] == 'source')
    bank.update(pearson_spectral=np.c_[bank['pearson'], bank['spectral']],
                mean_spi=B.mean_spi(bank['aligned'], bank['aligned'].shape[1] // (M * (M - 1))),
                catch22=B.catch22(x), tangent=B.tangent(x, source))
    del bank['aligned']
    return bank, source


def main(root, output):
    config = yaml.safe_load(fit_source.SPEC.read_text())
    bank, source = load(root)
    held = np.flatnonzero(bank['role'] != 'source')
    train = {k: v[source] for k, v in bank.items()}
    test = {k: v[held] for k, v in bank.items()}
    predictions, selection = {}, {}
    for name, (view, caps) in METHODS.items():
        fit_source.METHODS[name] = (view, 'pca')
        fitted, report = fit_source.fit_fmri(train, name, config if caps is None else {**config, 'pca_caps': caps})
        mapped, _ = fit_source.fmri_view(test, fitted['original_view'])
        predictions[name] = np.clip(fitted['model'].predict(fitted['transform'].transform(mapped, np.arange(len(held)))).reshape(-1), 0, 1)
        selection[name] = dict(selected=report['selected'], source_validation_brier=report['selected_source_validation_brier'])
        print('source selection', name, selection[name], flush=True)
    for kind in ('statistical', 'neural', 'minirocket'):
        saved = json.loads((root / f'predictions-{kind}.json').read_text())
        position = {r: k for k, r in enumerate(saved['row_id'])}
        order = [position[r] for r in test['row_id'].tolist()]
        predictions.update({n: np.array(v)[order] for n, v in saved['probability'].items()})
    result = {}
    for role in ROLES:
        ix = test['role'] == role
        y, groups = test['y'][ix], test['group'][ix]
        people = sorted(set(groups))
        draws = np.random.default_rng(261031).integers(0, len(people), (5000, len(people)))
        per_group = {n: group_metrics(y, p[ix], groups) for n, p in predictions.items()}
        methods = {n: dict({m: float(np.mean([v[g][m] for g in people])) for m in ('brier', 'balanced_accuracy', 'auroc')},
                           auroc_pooled=float(roc_auc_score(y, predictions[n][ix]))) for n, v in per_group.items()}
        contrasts = {}
        for a, b in PAIRS:
            entry = {}
            for metric in ('brier', 'balanced_accuracy', 'auroc'):
                sign = -1 if metric == 'brier' else 1
                v = sign * np.array([per_group[a][g][metric] - per_group[b][g][metric] for g in people])
                entry[metric] = dict(improvement=float(v.mean()), conditional_95_interval=np.quantile(v[draws].mean(1), [.025, .975]).tolist())
            contrasts[f'{a} versus {b}'] = entry
        result[role] = dict(recordings=int(ix.sum()), participants=len(people), methods=methods, contrasts=contrasts)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, dict(roles=result, source_selection=selection, new_methods=sorted(METHODS),
        probability={n: predictions[n].tolist() for n in METHODS}, row_id=test['row_id'].tolist(),
        bank_sha256=file_sha256(root / 'bank.npz'),
        scope='Post-confirmation comparator fill: source-only selection under the fMRI protocol; confirmation outcomes were already known, so these rows are exploratory.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    main(a.root, a.output)
