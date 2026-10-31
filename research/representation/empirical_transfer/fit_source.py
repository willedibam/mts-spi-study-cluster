"""Fit source-only statistical models; confirmation labels are never scored here."""
import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from mtsspi.provenance import atomic_json, file_sha256
from mtsspi.representation_screen import fit_view
from research.representation.shared.interaction_share_learning import ridge_readout
from research.representation.neurotycho.neurotycho_statistical import fit_grouped
from research.representation.shared.neurotycho_learning import balanced_brier, date_state_weights
from research.representation.empirical_transfer.splits import source_folds, group_brier

SPEC = Path('research/representation/neurotycho/configs/analysis/neurotycho-transfer-260910.yaml')
METHODS = {'m-pca': ('m', 'pca'), 'z-pca': ('z', 'pca'), 'mz-pca': ('m+z', 'pca'),
           'm-rbf': ('m', 'rbf'), 'pearson': ('pearson', 'pca'),
           'spectral': ('spectral', 'pca'), 'spectral-pooled': ('spectral_pooled', 'pca'),
           'aligned': ('aligned', 'pca'), 'validity': ('validity', 'pca'),
           'pearson-z': ('pearson+z', 'pca'),
           'single-spi': ('u', 'pca'), 'single-spi-z': ('u+z', 'pca')}


@dataclass
class SeparateProjection:
    """Keep baseline directions intact when appending a separate z projection."""
    baseline: object
    z: object

    def transform(self, bank, indices):
        return np.concatenate([self.baseline.transform(bank, indices), self.z.transform(bank, indices)], axis=1)


def fmri_view(bank, view):
    """Give an ordinary-feature block the established standardized 'u' treatment."""
    names = view.split('+')
    ordinary = [name for name in names if name not in ('m', 'z', 'validity')]
    if len(ordinary) > 1:
        raise ValueError('only one ordinary feature block is supported per view')
    if ordinary:
        bank = {**bank, 'u': bank[ordinary[0]]}
        names = ['u' if name == ordinary[0] else name for name in names]
    return bank, '+'.join(names)


def fit_fmri(bank, method, config):
    view, head = METHODS[method]
    bank, view = fmri_view(bank, view)
    choices = [dict(components=k, alpha=a) for k, a in itertools.product(
        config['pca_caps'], sorted(config['ridge_alpha_grid'], reverse=True))]
    reports = [dict(choice=choice, folds=[]) for choice in choices]
    for train, valid in source_folds(bank['group']):
        cache = {}
        weights = date_state_weights(bank['group'][train], bank['y'][train])
        for report in reports:
            choice = report['choice']
            k = choice['components']
            if k not in cache:
                transform, x = fit_view(bank, view, train, {**config['preprocessing'], 'pca_dimensions': k})
                cache[k] = x, transform.transform(bank, valid)
            x, vx = cache[k]
            model = ridge_readout(x, bank['y'][train], head, choice['alpha'])
            # Refit with equal mass per source person/state. The kernel wrapper
            # forwards sample_weight to KernelRidge.
            model.fit(x, bank['y'][train], sample_weight=weights)
            p = np.clip(model.predict(vx).reshape(-1), 0, 1)
            report['folds'].append(dict(training_ids=bank['row_id'][train].tolist(),
                validation_ids=bank['row_id'][valid].tolist(), y=bank['y'][valid].tolist(), probability=p.tolist(),
                person_balanced_brier=group_brier(bank['y'][valid], p, bank['group'][valid])))
    for report in reports:
        report['mean_brier'] = float(np.mean([f['person_balanced_brier'] for f in report['folds']]))
    selected = min(reports, key=lambda r: r['mean_brier'])
    choice = selected['choice']
    transform, x = fit_view(bank, view, np.arange(len(bank['y'])),
                            {**config['preprocessing'], 'pca_dimensions': choice['components']})
    model = ridge_readout(x, bank['y'], head, choice['alpha'])
    model.fit(x, bank['y'], sample_weight=date_state_weights(bank['group'], bank['y']))
    return dict(model=model, transform=transform, view=view, original_view=METHODS[method][0]), dict(
        selected=choice, selected_source_validation_brier=selected['mean_brier'], candidates=reports)


def main(args):
    config = yaml.safe_load(SPEC.read_text())
    root = args.root
    output = root / 'source-models'
    output.mkdir(exist_ok=True)
    name = args.method + (f'-{args.animal}' if args.animal else '')
    if (output / f'{name}.joblib').exists():
        raise FileExistsError(output / f'{name}.joblib')
    manifest = json.loads((root / 'manifest.json').read_text())
    bank_digest = file_sha256(root / 'bank.npz')
    if bank_digest != json.loads((root / 'bank.json').read_text())['sha256']:
        raise ValueError('bank checksum mismatch')
    with np.load(root / 'bank.npz', allow_pickle=False) as raw:
        keep = raw['role'] == 'source'
        if args.animal:
            keep &= raw['group'] != args.animal
        # Filter every fitted block before giving data to a fitting function.
        keys = {'y', 'M', 'T', 'row_id', 'group', 'role'}
        keys.update({'m', 'g', 'z', 'validity'} if args.animal else METHODS[args.method][0].split('+'))
        bank = {k: raw[k][keep] for k in keys}
    if args.animal:
        lookup = {r['row_id']: r for r in manifest['rows']}
        bank.update(animal=bank['group'], record_id=bank['row_id'],
                    archive=np.array([lookup[r]['archive'] for r in bank['row_id']]))
        fitted, report = fit_grouped(bank, args.method, config)
    else:
        fitted, report = fit_fmri(bank, args.method, config)
    model_path = output / f'{name}.joblib'
    joblib.dump(fitted, model_path)
    report.update(method=args.method, excluded_animal=args.animal,
                  source_rows=len(bank['y']), source_groups=sorted(set(bank['group'])),
                  target_data_used=False, bank_sha256=bank_digest,
                  model_sha256=file_sha256(model_path), config_sha256=file_sha256(SPEC),
                  source_runner_sha256=file_sha256(Path(__file__)))
    atomic_json(output / f'{name}.json', report)
    print(name, report['selected'], report['selected_source_validation_brier'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--animal', choices=['Chibi', 'George'])
    main(parser.parse_args())
