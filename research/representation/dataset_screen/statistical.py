"""One fixed statistical discovery panel; never reads reserved confirmation data."""
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
from research.representation.dataset_screen.common import folds, weights, probabilities, metrics

CONFIG = Path('research/representation/dataset_screen/protocol.yaml')
PREPROCESS = dict(minimum_valid_fraction=.95, variance_threshold=1e-8, z_scaling='center',
                  clip_standard_deviations=5, pca_solver='randomized', pca_random_state=1729)


def load_bank(root):
    if file_sha256(root / 'bank.npz') != json.loads((root / 'bank.json').read_text())['sha256']:
        raise ValueError('bank checksum mismatch')
    audit = json.loads((root / 'input-audit.json').read_text())
    if file_sha256(root / 'univariate.npz') != audit['univariate_sha256']:
        raise ValueError('amplitude/spectral baseline checksum mismatch')
    with np.load(root / 'bank.npz', allow_pickle=False) as source:
        bank = {k: source[k] for k in ['m', 'z', 'validity', 'pearson', 'aligned', 'y', 'group', 'role', 'row_id']}
        bank['spi_order'] = source['spi_order']
    with np.load(root / 'univariate.npz', allow_pickle=False) as source:
        if not np.array_equal(source['row_id'], bank['row_id']):
            raise ValueError('baseline row order differs')
        bank['u'] = source['u']
    if not set(bank['role']) <= {'source', 'screen'}:
        raise ValueError('reserved confirmation entered discovery')
    return bank


def source_rows(bank):
    ix = bank['role'] == 'source'
    return {k: v[ix] for k, v in bank.items() if k != 'spi_order'}


def feature_bank(bank, name, spi_index=None, spi_count=None):
    if name == 'single_spi':
        edges = bank['aligned'].shape[1] // spi_count
        return {**bank, 'u': bank['aligned'][:, spi_index * edges:(spi_index + 1) * edges]}, 'u'
    if name in ['pearson', 'aligned']:
        return {**bank, 'u': bank[name]}, 'u'
    if name in ['m', 'm_rbf']:
        return bank, 'm'
    if name in ['univariate', 'univariate_rbf']:
        return bank, 'u'
    return bank, name


@dataclass
class Blocks:
    transforms: list
    factors: list

    def transform(self, bank, ix):
        return np.concatenate([scale * transform.transform(bank, ix)
                               for scale, transform in zip(self.factors, self.transforms)], axis=1)


def fit_projection(bank, view, ix, cap, add_z=False, z_factor=1.):
    first, x = fit_view(bank, view, ix, {**PREPROCESS, 'pca_dimensions': cap})
    if not add_z:
        return first, x
    second, z = fit_view(bank, 'z', ix, {**PREPROCESS, 'pca_dimensions': cap})
    return Blocks([first, second], [1., z_factor]), np.c_[x, z_factor * z]


def fit_one(source, name, settings, spi_index=None, spi_count=None, add_z=False):
    bank, view = feature_bank(source, name, spi_index, spi_count)
    head = 'rbf' if name.endswith('_rbf') else 'pca'
    factors = [0., .25, 1.] if add_z else [1.]
    choices = [dict(cap=k, alpha=a, z_factor=f) for k, a, f in itertools.product(
        settings['pca_caps'], sorted(settings['ridge_alphas'], reverse=True), factors)]
    reports = [dict(choice=choice, folds=[]) for choice in choices]
    targets = np.eye(int(source['y'].max()) + 1)[source['y']]
    for train, valid in folds(source['y'], source['group']):
        cache = {}
        for report in reports:
            choice = report['choice']; key = choice['cap'], choice['z_factor']
            if key not in cache:
                transform, x = fit_projection(bank, view, train, key[0], add_z, key[1])
                cache[key] = x, transform.transform(bank, valid)
            x, vx = cache[key]
            model = ridge_readout(x, targets[train], head, choice['alpha'])
            model.fit(x, targets[train], sample_weight=weights(source['y'][train], source['group'][train]))
            p = probabilities(model.predict(vx))
            report['folds'].append(dict(training_ids=source['row_id'][train].tolist(),
                validation_ids=source['row_id'][valid].tolist(), probability=p.tolist(),
                **metrics(source['y'][valid], p, source['group'][valid])))
    for report in reports:
        report['mean_brier'] = float(np.mean([f['brier'] for f in report['folds']]))
    selected = min(reports, key=lambda r: r['mean_brier'])
    choice = selected['choice']; ix = np.arange(len(source['y']))
    transform, x = fit_projection(bank, view, ix, choice['cap'], add_z, choice['z_factor'])
    model = ridge_readout(x, targets, head, choice['alpha'])
    model.fit(x, targets, sample_weight=weights(source['y'], source['group']))
    fitted = dict(transform=transform, model=model, name=name, spi_index=spi_index, spi_count=spi_count)
    return fitted, dict(selected=choice, source_validation_brier=selected['mean_brier'], candidates=reports,
                        source_groups=sorted(set(source['group'])), source_rows=source['row_id'].tolist())


def predict(fitted, bank, ix):
    mapped, _ = feature_bank(bank, fitted['name'], fitted['spi_index'], fitted['spi_count'])
    return probabilities(fitted['model'].predict(fitted['transform'].transform(mapped, ix)))


def main(root):
    # Stable module identity for saved projection objects under python -m.
    from research.representation.dataset_screen import statistical as implementation
    bank = load_bank(root); source = source_rows(bank)
    settings = yaml.safe_load(CONFIG.read_text())['comparisons']
    output = root / 'screen-models'; output.mkdir(exist_ok=True)
    if any((output / f'{name}.joblib').exists() for name in ['m', 'z', 'baseline_plus_z']):
        raise FileExistsError('statistical discovery models already exist')
    fits, reports = {}, {}
    for method in ['univariate', 'univariate_rbf', 'pearson', 'aligned', 'm', 'm_rbf', 'z', 'validity']:
        fits[method], reports[method] = implementation.fit_one(source, method, settings)
        print('Source selection', method, reports[method]['source_validation_brier'], flush=True)
    best, catalogue = None, []
    for index, name in enumerate(bank['spi_order']):
        fitted, report = implementation.fit_one(source, 'single_spi', settings, index, len(bank['spi_order']))
        catalogue.append(dict(spi_index=index, name=str(name), selected=report['selected'], source_validation_brier=report['source_validation_brier']))
        if best is None or report['source_validation_brier'] < best[1]['source_validation_brier']:
            best = fitted, report
        if (index + 1) % 32 == 0: print('Individual SPI screen', index + 1, flush=True)
    fits['single_spi'], reports['single_spi'] = best
    baseline = min(['univariate', 'univariate_rbf', 'pearson', 'aligned', 'm', 'm_rbf', 'single_spi'],
                   key=lambda name: reports[name]['source_validation_brier'])
    fits['m_plus_z'], reports['m_plus_z'] = implementation.fit_one(source, 'm', settings, add_z=True)
    f = fits[baseline]
    fits['baseline_plus_z'], reports['baseline_plus_z'] = implementation.fit_one(source, baseline, settings,
        f['spi_index'], f['spi_count'], add_z=True)
    # The z-factor zero candidate includes the unaugmented representation; no
    # favourable combination is selected using discovery-screen outcomes.
    receipts = {}
    for name in fits:
        path = output / f'{name}.joblib'
        joblib.dump(fits[name], path)
        reports[name].update(model_sha256=file_sha256(path), target_data_used=False)
        atomic_json(output / f'{name}.json', reports[name])
        receipts[name] = reports[name]['model_sha256']
    atomic_json(root / 'statistical-source-lock.json', dict(models=receipts, selected_baseline=baseline,
        selected_spi=str(bank['spi_order'][fits['single_spi']['spi_index']]), catalogue_selection=catalogue,
        bank_sha256=file_sha256(root / 'bank.npz'), protocol_sha256=file_sha256(CONFIG),
        source_runner_sha256=file_sha256(Path(__file__)), stage='discovery; reserved confirmation absent'))
    ix = np.flatnonzero(bank['role'] == 'screen')
    outcomes = {name: dict(**metrics(bank['y'][ix], p := predict(fit, bank, ix), bank['group'][ix]),
                          probability=p.tolist()) for name, fit in fits.items()}
    atomic_json(root / 'statistical-screen.json', dict(methods=outcomes, row_id=bank['row_id'][ix].tolist(),
        y=bank['y'][ix].tolist(), group=bank['group'][ix].tolist(), selected_baseline=baseline,
        source_lock_sha256=file_sha256(root / 'statistical-source-lock.json'),
        stage='Exploratory dataset screen; not independent confirmation or proof of physiological mechanism.'))
    print('Discovery screen complete; selected source baseline', baseline, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
