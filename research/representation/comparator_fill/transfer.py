"""Comparator ladder on the discovery-screen banks (CHB-MIT, HMC, GRABMyo, LANL), under their frozen protocol.

Protocol (unchanged from dataset_screen.statistical): every model is selected on source rows only (three grouped
folds, PCA cap and ridge penalty by source Brier, z appended as a separate projection with a source-chosen
weight that includes zero) and scored once on the screen rows at the argmax decision. With --siena the CHB-MIT
fits are applied unchanged to the Siena confirmation bank; those rows are post hoc, since Siena outcomes are known.

Added here: r + z, per-channel log spectra and spectra + z, per-channel univariate + z, r + spectra (+ z),
mean-SPI (+ z), tangent-space covariance, per-channel catch22, and z read through its first PC or at full rank.
The tangent reference is fitted on source rows. Pooled AUROC is added for these and for the existing methods.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from mtsspi.provenance import atomic_json, file_sha256
from research.representation.comparator_fill import blocks as B
from research.representation.dataset_screen import statistical
from research.representation.dataset_screen.common import metrics
from research.representation.dataset_screen.review import contrast, grouped_scores

# name -> (block treated as the standardized baseline view, append z, PCA caps override)
METHODS = dict(pearson_z=('pearson', True, None), spectra=('spectra', False, None), spectra_z=('spectra', True, None),
               univariate_z=('u', True, None), pearson_spectra=('pearson_spectra', False, None),
               pearson_spectra_z=('pearson_spectra', True, None), mean_spi=('mean_spi', False, None),
               mean_spi_z=('mean_spi', True, None), tangent=('tangent', False, None), catch22=('catch22', False, None),
               z_pc1=('z', False, [1]), z_full=('z', False, [10 ** 6]))
PAIRS = [('pearson_z', 'pearson'), ('spectra_z', 'spectra'), ('univariate_z', 'univariate'),
         ('pearson_spectra_z', 'pearson_spectra'), ('mean_spi_z', 'mean_spi'), ('z', 'pearson'), ('z', 'mean_spi'),
         ('z', 'spectra'), ('z', 'tangent'), ('z', 'catch22'), ('z_full', 'z'), ('z_pc1', 'z')]


def raw_windows(root, row_ids):
    with np.load(root / 'raw-views.npz', allow_pickle=False) as raw:
        return np.array([raw[str(r)] for r in row_ids], dtype=np.float64)


def extend(bank, x, train):
    """Add the new blocks; x is (n, T, M) in bank row order and train indexes the rows the tangent reference may see."""
    M = x.shape[2]
    spectra = bank['u'][:, 11 * M:43 * M]
    if bank['u'].shape[1] != 75 * M:
        raise ValueError('univariate layout differs from dataset_screen.prepare.univariate')
    return {**bank, 'spectra': spectra, 'pearson_spectra': np.c_[bank['pearson'], spectra],
            'mean_spi': B.mean_spi(bank['aligned'], len(bank['spi_order'])), 'catch22': B.catch22(x),
            'tangent': B.tangent(x, train)}


def view(bank, block):
    return bank if block in ('u', 'z') else {**bank, 'u': bank[block]}


def fit(source, name, settings):
    block, add_z, caps = METHODS[name]
    settings = settings if caps is None else {**settings, 'pca_caps': caps}
    fitted, report = statistical.fit_one(view(source, block), 'z' if block == 'z' else 'univariate', settings, add_z=add_z)
    return fitted, dict(selected=report['selected'], source_validation_brier=report['source_validation_brier'])


def predict(fitted, bank, name, ix):
    return statistical.predict(fitted, view(bank, METHODS[name][0]), ix)


def auroc(y, p):
    y = np.asarray(y)
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    return float(roc_auc_score(y, p, multi_class='ovr', average='macro'))


def score(y, groups, predictions, existing):
    """Metrics for the new methods, pooled AUROC for all, and paired contrasts against the stored predictions."""
    everything = {**existing, **predictions}
    methods = {n: dict(**metrics(y, p, groups), auroc_pooled=auroc(y, p), probability=p.tolist())
               for n, p in predictions.items()}
    per_group = {n: grouped_scores(y, p, groups) for n, p in everything.items()}
    contrasts = {f'{a} versus {b}': contrast(per_group[a][0], per_group[b][0], per_group[a][1])
                 for a, b in PAIRS if a in everything and b in everything}
    return dict(methods=methods, existing_auroc_pooled={n: auroc(y, p) for n, p in existing.items()}, contrasts=contrasts)


def stored(path, y, row_ids):
    """Predictions already scored for this bank, keyed by method and put in the bank's row order."""
    saved = json.loads(path.read_text())
    position = {r: k for k, r in enumerate(saved['row_id'])}
    if set(position) != set(row_ids) or len(position) != len(row_ids):
        raise ValueError(f'{path.name} rows differ from the bank')
    order = [position[r] for r in row_ids]
    if [saved['y'][k] for k in order] != list(y):
        raise ValueError(f'{path.name} labels differ from the bank')
    if 'methods' in saved:
        found = {n: v['probability'] for n, v in saved['methods'].items()}
    elif isinstance(saved['probability'], dict):
        found = saved['probability']
    else:
        found = {path.name.removesuffix('-screen.json'): saved['probability']}
    return {n: np.array(v)[order] for n, v in found.items()}


def main(root, siena, output):
    bank = statistical.load_bank(root)
    x = raw_windows(root, bank['row_id'])
    source_ix = np.flatnonzero(bank['role'] == 'source')
    screen_ix = np.flatnonzero(bank['role'] == 'screen')
    bank = extend(bank, x, source_ix)
    source = statistical.source_rows(bank)
    settings = yaml.safe_load(statistical.CONFIG.read_text())['comparisons']
    fits, selection = {}, {}
    for name in METHODS:
        fits[name], selection[name] = fit(source, name, settings)
        print('source selection', name, selection[name], flush=True)
    y, groups, rows = bank['y'][screen_ix], bank['group'][screen_ix], bank['row_id'][screen_ix].tolist()
    existing = stored(root / 'statistical-screen.json', y.tolist(), rows)
    for extra in sorted(root.glob('*-screen.json')):
        if extra.name != 'statistical-screen.json':
            existing.update(stored(extra, y.tolist(), rows))
    report = dict(dataset=root.name, bank_sha256=file_sha256(root / 'bank.npz'), source_selection=selection,
                  screen=score(y, groups, {n: predict(f, bank, n, screen_ix) for n, f in fits.items()}, existing),
                  scope='Post-discovery comparator fill: source-only selection under the discovery protocol; exploratory.')
    if siena is not None:
        from research.representation.dataset_screen.confirm_siena import inputs
        _, _, target = inputs(siena)
        tx = raw_windows(siena, target['row_id'])
        # The tangent reference stays the CHB-MIT source mean: stack source windows so the same reference is used.
        both = np.concatenate([x[source_ix], tx])
        target = extend(target, tx, np.arange(len(tx)))
        target['tangent'] = B.tangent(both, np.arange(len(source_ix)))[len(source_ix):]
        ix = np.arange(len(target['y']))
        existing = {}
        for kind in ('statistical', 'neural', 'minirocket'):
            existing.update(stored(siena / f'predictions-{kind}.json', target['y'].tolist(), target['row_id'].tolist()))
        report['siena'] = score(target['y'], target['group'], {n: predict(f, target, n, ix) for n, f in fits.items()}, existing)
        report['siena']['scope'] = 'CHB-MIT source fits applied unchanged; post hoc, because Siena outcomes were already known.'
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--siena', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    main(a.root, a.siena, a.output)
