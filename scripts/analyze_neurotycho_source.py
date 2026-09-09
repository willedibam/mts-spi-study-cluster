"""Run the declared source waveform/QC and grouped spectral-baseline pilot."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.neurotycho_pilot import preprocess, spectral_features, window_specs


def numeric_mat(path, key):
    data = loadmat(path, simplify_cells=True)
    value = np.asarray(data[key])
    if value.ndim != 1:
        raise ValueError(f'expected vector in {path}')
    return value


def build_archive(root, animal, montage, output):
    complete = json.loads((root / 'complete.json').read_text())
    channels = complete['channels']
    indices = [[channels.index(i), channels.index(j)] for i, j in montage['pairs']]
    records, matrices, features = [], [], []
    provenance = {}
    for session in complete['sessions']:
        directory = root / session
        condition = loadmat(directory / 'Condition.mat', simplify_cells=True)
        specs = window_specs(condition)
        time = numeric_mat(directory / 'ECoGTime.mat', 'ECoGTime')
        if abs(time[0]) > 1e-8 or not np.allclose(np.diff(time), .001, rtol=0, atol=1e-8):
            raise ValueError('unexpected time origin or nonuniform timestamps')
        chunks = np.empty((len(specs), len(channels), 28000), dtype=np.float32)
        for column, channel in enumerate(channels):
            path = directory / f'ECoG_ch{channel}.mat'
            waveform = numeric_mat(path, f'ECoGData_ch{channel}')
            if len(waveform) != len(time):
                raise ValueError('signal/time length mismatch')
            provenance[str(path)] = json.loads(path.with_suffix('.mat.json').read_text())['sha256']
            for i, spec in enumerate(specs):
                lo, hi = spec['context_start'], spec['context_stop']
                if lo < 0 or hi > len(waveform):
                    raise ValueError('context outside acquisition')
                chunks[i, column] = waveform[lo:hi]
        for spec, chunk in zip(specs, chunks):
            x, quality = preprocess(chunk, indices)
            record = dict(spec, animal=animal, archive=root.name, session=session, quality=quality)
            if x is not None:
                record['array_row'] = len(matrices)
                matrices.append(x); features.append(spectral_features(x))
            records.append(record)
        del chunks
    accepted = [r for r in records if r['quality']['accepted']]
    counts = [sum(r['target'] == label for r in accepted) for label in [0, 1]]
    usable = min(counts) >= 12
    np.savez_compressed(output / (root.name + '.npz'), x=np.array(matrices), spectral=np.array(features),
                        y=np.array([r['target'] for r in accepted]))
    metadata = dict(animal=animal, archive=root.name, usable=usable, counts=counts, records=records,
                    waveform_hashes=provenance, montage=montage)
    (output / (root.name + '.json')).write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{animal}: accepted{counts}, usable={usable}', flush=True)
    return metadata


def fit(x, y, groups, c):
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          LogisticRegression(C=c, max_iter=2000, solver='lbfgs'))
    # Equal animal/state contribution, even when archives or QC counts differ.
    weight = np.zeros(len(y))
    for animal in np.unique(groups):
        for label in [0, 1]:
            select = (groups == animal) & (y == label)
            weight[select] = 1 / select.sum()
    weight *= len(y) / weight.sum()
    model.fit(x, y, logisticregression__sample_weight=weight)
    return model


def baseline(output, metadata):
    xs, ys, groups = [], [], []
    for meta in metadata:
        if not meta['usable']:
            continue
        bank = np.load(output / (meta['archive'] + '.npz'))
        xs.append(bank['spectral']); ys.append(bank['y'])
        groups.extend([meta['animal']] * len(bank['y']))
    x, y, groups = np.concatenate(xs), np.concatenate(ys), np.array(groups)
    if len(np.unique(groups)) != 4:
        raise ValueError('four-animal gate not passed; preserve QC, do not fit incomplete scout')
    reports = []
    predictions = np.full(len(y), np.nan)
    for animal in np.unique(groups):
        train, test = groups != animal, groups == animal
        choices = []
        for c in [.01, .1, 1., 10.]:
            scores = []
            for inner in np.unique(groups[train]):
                inner_train, valid = train & (groups != inner), train & (groups == inner)
                model = fit(x[inner_train], y[inner_train], groups[inner_train], c)
                scores.append(balanced_accuracy_score(y[valid], model.predict(x[valid])))
            choices.append(dict(c=c, scores=scores, mean=float(np.mean(scores))))
        selected = max(choices, key=lambda v: v['mean'])
        model = fit(x[train], y[train], groups[train], selected['c'])
        p = model.predict_proba(x[test])[:, 1]
        predictions[test] = p
        reports.append(dict(animal=animal, n_test=int(test.sum()), selected_c=selected['c'],
            balanced_accuracy=float(balanced_accuracy_score(y[test], p >= .5)),
            auroc=float(roc_auc_score(y[test], p)), brier=float(brier_score_loss(y[test], p)),
            source_only_cv=choices))
    np.savez_compressed(output / 'spectral-predictions.npz', y=y, probability=predictions, animal=groups)
    report = dict(protocol='docs/neurotycho-source-pilot.md', unit='four held-out animals; exploratory',
        animals=reports, mean_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in reports])),
        mean_auroc=float(np.mean([r['auroc'] for r in reports])))
    (output / 'spectral-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.data / args.plan).read_text())
    metadata = []
    for name in plan['source_names']:
        root = args.data / name
        meta_path = args.output / (name + '.json')
        if meta_path.exists():
            metadata.append(json.loads(meta_path.read_text()))
            continue
        complete = json.loads((root / 'complete.json').read_text())
        metadata.append(build_archive(root, complete['animal'], plan['montages'][complete['animal']], args.output))
    if not args.qc_only:
        baseline(args.output, metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/neurotycho_source_260910'))
    parser.add_argument('--output', type=Path, default=Path('results/neurotycho_source_pilot_260910'))
    parser.add_argument('--plan', default='scout-plan.json')
    parser.add_argument('--qc-only', action='store_true')
    main(parser.parse_args())
