"""Freeze source decisions, then evaluate existing fMRI models without refitting."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from mtsspi.provenance import atomic_json, file_sha256

STATISTICAL = ['m-pca', 'z-pca', 'mz-pca', 'm-rbf', 'pearson', 'pearson-z',
               'spectral', 'spectral-pooled', 'aligned', 'validity', 'single-spi', 'single-spi-z']
RAW = ['aligned-neural', 'inceptiontime', 'minirocket']
ROLES = ['development_shift', 'confirmation_subject_only', 'confirmation_primary']
CONTRASTS = [('mz-pca', 'm-pca'), ('pearson-z', 'pearson'), ('single-spi-z', 'single-spi')]


def freeze(root):
    path = root / 'evaluation-lock.json'
    if path.exists():
        raise FileExistsError(path)
    files = ['manifest.json', 'bank.json', 'source-review.json', 'single-spi-selection.json']
    reports = {}
    for method in STATISTICAL + RAW:
        relative = f'source-models/{method}.json'
        report = json.loads((root / relative).read_text())
        if report.get('target_data_used', report.get('identity', {}).get('target_data_used')) is not False:
            raise ValueError('source-only provenance absent')
        files.append(relative)
        if method in RAW:
            members = [(m['path'], m['sha256']) for m in report['members']]
        else:
            members = [(f'{method}.joblib', report['model_sha256'])]
        for name, digest in members:
            p = root / 'source-models' / name
            if file_sha256(p) != digest:
                raise ValueError('model checksum mismatch before lock')
            files.append(f'source-models/{name}')
        selected = report.get('selected', report.get('selected_C'))
        if isinstance(selected, dict):
            selected = {k: v for k, v in selected.items() if k != 'folds'}
        reports[method] = dict(selected=selected,
                               spi_name=report.get('spi_name'))
    atomic_json(path, dict(files={f: file_sha256(root / f) for f in files},
        methods=STATISTICAL + RAW, decisions=reports, primary_role='confirmation_primary',
        primary_contrast=['mz-pca', 'm-pca'], secondary_contrasts=CONTRASTS[1:], threshold=.5,
        uncertainty='Paired whole-participant bootstrap, conditional on these fitted models; nine confirmation people.',
        interpretation='Source-nominated panels are exploratory descriptions; waveform and estimator checks remain required. No biological mechanism follows from a predictive gain.',
        evaluation_runner_sha256=file_sha256(Path(__file__))))


def checked_bank(root):
    lock = json.loads((root / 'evaluation-lock.json').read_text())
    if lock['evaluation_runner_sha256'] != file_sha256(Path(__file__)):
        raise ValueError('evaluation implementation changed after lock')
    for name, digest in lock['files'].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f'frozen input changed: {name}')
    if file_sha256(root / 'bank.npz') != json.loads((root / 'bank.json').read_text())['sha256']:
        raise ValueError('feature bank checksum mismatch')
    with np.load(root / 'bank.npz', allow_pickle=False) as archive:
        keep = archive['role'] != 'source'
        keys = ['x', 'm', 'z', 'validity', 'aligned', 'pearson', 'spectral', 'spectral_pooled',
                'row_id', 'group', 'role', 'y']
        bank = {key: archive[key][keep] for key in keys}
    if set(bank['role']) != set(ROLES):
        raise ValueError('unexpected confirmation/development roles')
    return bank


def predict_statistical(root, bank):
    from research.representation.empirical_transfer.fit_source import fmri_view
    output = {}
    indices = np.arange(len(bank['y']))
    for method in STATISTICAL:
        fitted = joblib.load(root / 'source-models' / f'{method}.joblib')
        view_bank = bank
        if method.startswith('single-spi'):
            start = fitted['spi_index'] * fitted['edges_per_spi']
            view_bank = {**bank, 'u': bank['aligned'][:, start:start + fitted['edges_per_spi']]}
        mapped, _ = fmri_view(view_bank, fitted['original_view'])
        output[method] = np.clip(fitted['model'].predict(
            fitted['transform'].transform(mapped, indices)).reshape(-1), 0, 1)
    return output


def predict_neural(root, bank):
    import torch
    from research.representation.shared.neurotycho_learning import predict_binary
    from research.representation.shared.representation_state_neural import make_encoder
    from research.representation.learned_baselines.inceptiontime_baseline import InceptionNetwork
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    x = torch.as_tensor(bank['x'], dtype=torch.float32)
    output = {}
    for method in ['aligned-neural', 'inceptiontime']:
        report = json.loads((root / 'source-models' / f'{method}.json').read_text())
        members = []
        for member in report['members']:
            saved = torch.load(root / 'source-models' / member['path'], map_location='cpu', weights_only=True)
            model = InceptionNetwork(saved['spec']) if method == 'inceptiontime' else make_encoder(saved['spec'])
            model.load_state_dict(saved['state_dict'])
            members.append(predict_binary(model, x, batch=16))
        output[method] = np.mean(members, axis=0)
    return output


def predict_rocket(root, bank):
    from importlib.metadata import version
    report = json.loads((root / 'source-models/minirocket.json').read_text())
    for package, expected in report['identity']['versions'].items():
        if version(package) != expected:
            raise ValueError(f'MiniRocket runtime changed: {package}')
    predictions = []
    for member in report['members']:
        fitted = joblib.load(root / 'source-models' / member['path'])
        features = fitted['rocket'].transform(bank['x'].transpose(0, 2, 1))
        predictions.append(fitted['head'].predict_proba(features)[:, 1])
    return {'minirocket': np.mean(predictions, axis=0)}


def group_metrics(y, p, groups):
    result = {}
    for group in sorted(set(groups)):
        ix = groups == group
        if set(y[ix]) != {0, 1}:
            raise ValueError('participant lacks one state')
        result[str(group)] = dict(
            brier=float(np.mean([np.mean((p[ix & (y == label)] - label) ** 2) for label in (0, 1)])),
            balanced_accuracy=float(np.mean([np.mean((p[ix & (y == label)] >= .5) == label) for label in (0, 1)])),
            auroc=float(roc_auc_score(y[ix], p[ix])))
    return result


def summarize(root, bank):
    predictions = {}
    for kind in ['statistical', 'neural', 'minirocket']:
        saved = json.loads((root / f'predictions-{kind}.json').read_text())
        if saved['row_id'] != bank['row_id'].tolist() or saved['lock_sha256'] != file_sha256(root / 'evaluation-lock.json'):
            raise ValueError('prediction rows or lock differ')
        predictions.update({k: np.array(v) for k, v in saved['probability'].items()})
    if set(predictions) != set(STATISTICAL + RAW):
        raise ValueError('incomplete comparator panel')
    result = {}
    for role in ROLES:
        ix = bank['role'] == role
        people = sorted(set(bank['group'][ix]))
        draws = np.random.default_rng(261031).integers(0, len(people), (5000, len(people)))
        per_group = {m: group_metrics(bank['y'][ix], p[ix], bank['group'][ix]) for m, p in predictions.items()}
        methods, contrasts = {}, {}
        for method, values in per_group.items():
            methods[method] = dict(groups=values, metrics={})
            for metric in ['brier', 'balanced_accuracy', 'auroc']:
                v = np.array([values[p][metric] for p in people])
                methods[method]['metrics'][metric] = dict(mean=float(v.mean()),
                    conditional_95_interval=np.quantile(v[draws].mean(1), [.025, .975]).tolist())
        for combined, base in CONTRASTS:
            contrasts[f'{combined} versus {base}'] = {}
            for metric in ['brier', 'balanced_accuracy', 'auroc']:
                sign = -1 if metric == 'brier' else 1
                v = sign * np.array([per_group[combined][p][metric] - per_group[base][p][metric] for p in people])
                contrasts[f'{combined} versus {base}'][metric] = dict(improvement=float(v.mean()),
                    conditional_95_interval=np.quantile(v[draws].mean(1), [.025, .975]).tolist())
        result[role] = dict(recordings=int(ix.sum()), participants=len(people), methods=methods, contrasts=contrasts)
    atomic_json(root / 'evaluation-summary.json', dict(roles=result,
        lock_sha256=file_sha256(root / 'evaluation-lock.json'),
        scope='Fixed-source models, threshold 0.5, equally weighted participant/state metrics. Bootstrap covers target participants conditional on models, not source-training or model-selection uncertainty.'))


def main(root, mode):
    if mode == 'freeze':
        freeze(root)
        return
    output = root / ('evaluation-summary.json' if mode == 'summarize' else f'predictions-{mode}.json')
    if output.exists():
        raise FileExistsError(output)
    bank = checked_bank(root)
    if mode == 'summarize':
        summarize(root, bank)
        return
    probability = {'statistical': predict_statistical, 'neural': predict_neural, 'minirocket': predict_rocket}[mode](root, bank)
    for values in probability.values():
        if values.shape != bank['y'].shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('invalid predictions')
    atomic_json(output, dict(probability={k: v.tolist() for k, v in probability.items()},
        row_id=bank['row_id'].tolist(), group=bank['group'].tolist(), role=bank['role'].tolist(), y=bank['y'].tolist(),
        lock_sha256=file_sha256(root / 'evaluation-lock.json'), fitted_on_target=False))
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=['freeze', 'statistical', 'neural', 'minirocket', 'summarize'])
    args = parser.parse_args()
    main(args.root, args.mode)
