"""Source-only fMRI raw baselines; common participant folds, no target scoring."""
import argparse
from importlib.metadata import version
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from mtsspi.provenance import atomic_json, file_sha256
from research.representation.empirical_transfer.splits import source_folds, group_brier, group_weights


def inputs(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    path = root / 'views.npz'
    if file_sha256(path) != manifest['archive_sha256']:
        raise ValueError('input checksum mismatch')
    rows = [r for r in manifest['rows'] if r['role'] == 'source']
    with np.load(path, allow_pickle=False) as data:
        x = np.array([data[r['row_id']] for r in rows], dtype=np.float32)
    return x, np.array([r['target'] for r in rows]), np.array([r['group'] for r in rows]), rows


def neural(args, x, y, groups, output, identity):
    import torch
    from research.representation.shared.neurotycho_learning import fit_binary, predict_binary
    from research.representation.learned_baselines.inceptiontime_baseline import fit_network
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.method == 'inceptiontime':
        config_path = Path('research/representation/learned_baselines/configs/analysis/inceptiontime-followup-260911.yaml')
        config = yaml.safe_load(config_path.read_text())
        spec = {**config['architecture'], 'input_channels': x.shape[-1]}
        training = config['training']
        seeds = config['member_seeds']
        fit = fit_network
    else:
        config_path = Path('research/representation/neurotycho/configs/analysis/neurotycho-transfer-260910.yaml')
        config = yaml.safe_load(config_path.read_text())
        spec, training, seeds = config['raw_encoder'], config['neural_training'], config['neural_seeds']
        fit = fit_binary
    identity.update(config_sha256=file_sha256(config_path), spec=spec, training=training,
                    seeds=seeds, torch=version('torch'), numpy=version('numpy'), device=args.device)
    tensor = torch.as_tensor(x, device=args.device)
    candidates = []
    for lr, decay in itertools.product(training['learning_rates'], training['weight_decays']):
        folds = []
        for fold, (train, valid) in enumerate(source_folds(groups)):
            model, report = fit(tensor[train], y[train], groups[train], spec, training, lr, decay, 11,
                                validation=(tensor[valid], y[valid]))
            p = predict_binary(model, tensor[valid], training['batch_size'])
            report.update(fold=fold, training_rows=train.tolist(), validation_rows=valid.tolist(),
                          participant_brier=group_brier(y[valid], p, groups[valid]), probability=p.tolist())
            folds.append(report)
            atomic_json(output / f'{args.method}-progress.json', dict(identity=identity,
                        completed_candidates=candidates, current=dict(lr=lr, decay=decay, folds=folds)))
            print(args.method, lr, decay, 'fold', fold, 'Brier', report['participant_brier'], flush=True)
            del model
        candidates.append(dict(lr=lr, decay=decay, folds=folds,
                               mean_brier=float(np.mean([f['participant_brier'] for f in folds]))))
    selected = min(candidates, key=lambda r: r['mean_brier'])
    epochs = int(np.median([f['best_epoch'] for f in selected['folds']]))
    members = []
    for seed in seeds:
        model, report = fit(tensor, y, groups, spec, training, selected['lr'], selected['decay'], seed, epochs=epochs)
        path = output / f'{args.method}-{seed}.pt'
        torch.save(dict(spec=spec, state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}), path)
        members.append(dict(seed=seed, report=report, path=path.name, sha256=file_sha256(path)))
        print('Saved', args.method, seed, 'epochs', epochs, flush=True)
    return dict(identity=identity, candidates=candidates, selected=selected,
                source_selected_epochs=epochs, members=members, prediction_rule='arithmetic mean of member probabilities')


def minirocket(args, x, y, groups, output, identity):
    from aeon.transformations.collection.convolution_based import MiniRocket
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    raw = x.transpose(0, 2, 1)
    choices = [.01, .1, 1., 10.]
    candidates = {c: [] for c in choices}
    for train, valid in source_folds(groups):
        rocket = MiniRocket(n_kernels=10000, max_dilations_per_kernel=32, n_jobs=args.threads, random_state=11)
        features = rocket.fit_transform(raw[train])
        held = rocket.transform(raw[valid])
        for c in choices:
            head = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
            head.fit(features, y[train], logisticregression__sample_weight=group_weights(groups[train], y[train]))
            p = head.predict_proba(held)[:, 1]
            candidates[c].append(dict(training_rows=train.tolist(), validation_rows=valid.tolist(),
                probability=p.tolist(), participant_brier=group_brier(y[valid], p, groups[valid])))
    selected = min(choices, key=lambda c: np.mean([f['participant_brier'] for f in candidates[c]]))
    members = []
    for seed in [11, 23, 47]:
        rocket = MiniRocket(n_kernels=10000, max_dilations_per_kernel=32, n_jobs=args.threads, random_state=seed)
        features = rocket.fit_transform(raw)
        head = make_pipeline(StandardScaler(), LogisticRegression(C=selected, max_iter=5000))
        head.fit(features, y, logisticregression__sample_weight=group_weights(groups, y))
        path = output / f'minirocket-{seed}.joblib'
        joblib.dump(dict(rocket=rocket, head=head), path)
        members.append(dict(seed=seed, path=path.name, sha256=file_sha256(path)))
    identity['versions'] = {p: version(p) for p in ['aeon', 'numpy', 'numba', 'scikit-learn']}
    return dict(identity=identity, candidates=candidates, selected_C=selected, members=members,
                prediction_rule='arithmetic mean of member probabilities')


def main(args):
    output = args.root / 'source-models'
    output.mkdir(exist_ok=True)
    if (output / f'{args.method}.json').exists():
        raise FileExistsError('completed raw fit already exists')
    x, y, groups, rows = inputs(args.root)
    identity = dict(method=args.method, manifest_sha256=file_sha256(args.root / 'manifest.json'),
                    runner_sha256=file_sha256(Path(__file__)), source_rows=[r['row_id'] for r in rows],
                    source_groups=sorted(set(groups)), target_data_used=False, augmentation=False,
                    source_validation='three fixed participant folds; selection seed 11')
    result = (minirocket if args.method == 'minirocket' else neural)(args, x, y, groups, output, identity)
    atomic_json(output / f'{args.method}.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method', choices=['aligned-neural', 'inceptiontime', 'minirocket'], required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    main(parser.parse_args())
