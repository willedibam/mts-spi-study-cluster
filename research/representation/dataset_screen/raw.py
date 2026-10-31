"""Matched grouped discovery comparisons on physical-unit waveform inputs."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import joblib
import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from mtsspi.provenance import atomic_json, file_sha256
from research.representation.dataset_screen.common import folds, weights, metrics, probabilities


def input_manifest(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    audit = json.loads((root / 'input-audit.json').read_text())
    if file_sha256(root / 'raw-views.npz') != audit['raw_views_sha256']:
        raise ValueError('raw waveform checksum mismatch')
    return manifest


def load_role(root, manifest, role):
    rows = [r for r in manifest['rows'] if r['role'] == role]
    with np.load(root / 'raw-views.npz', allow_pickle=False) as raw:
        x = np.array([raw[r['row_id']] for r in rows], dtype=np.float32)
    return x, np.array([r['target'] for r in rows]), np.array([r['group'] for r in rows]), rows


def normalization(x):
    mean = x.mean((0, 1), dtype=np.float64).astype(np.float32)
    scale = x.std((0, 1), dtype=np.float64).astype(np.float32)
    if not np.isfinite(scale).all() or np.any(scale <= 1e-12):
        raise ValueError('flat/nonfinite source channel')
    return mean, scale


def rocket(root, x, y, groups, identity, output):
    from aeon.transformations.collection.convolution_based import MiniRocket
    alpha_grid = [1e-3, 1e-2, 1e-1, 1., 10., 100., 1000.]
    target = np.eye(int(y.max()) + 1)[y]
    reports = [dict(alpha=a, folds=[]) for a in alpha_grid]
    for train, valid in folds(y, groups):
        mean, scale = normalization(x[train])
        transform = MiniRocket(n_kernels=10000, max_dilations_per_kernel=32, n_jobs=4, random_state=11)
        # Small ridge penalties on 10,000 float32 features can make the dual
        # solve numerically singular even though regularization is positive.
        features = transform.fit_transform(((x[train] - mean) / scale).transpose(0, 2, 1)).astype(np.float64)
        held = transform.transform(((x[valid] - mean) / scale).transpose(0, 2, 1)).astype(np.float64)
        scaler = StandardScaler().fit(features)
        features, held = scaler.transform(features), scaler.transform(held)
        for report in reports:
            model = Ridge(alpha=report['alpha']).fit(features, target[train], sample_weight=weights(y[train], groups[train]))
            p = probabilities(model.predict(held))
            report['folds'].append(dict(training_rows=train.tolist(), validation_rows=valid.tolist(),
                probability=p.tolist(), **metrics(y[valid], p, groups[valid])))
    for r in reports: r['mean_brier'] = float(np.mean([f['brier'] for f in r['folds']]))
    selected = min(reports, key=lambda r:r['mean_brier'])
    mean, scale = normalization(x)
    members = []
    for seed in [11, 23, 47]:
        transform = MiniRocket(n_kernels=10000, max_dilations_per_kernel=32, n_jobs=4, random_state=seed)
        features = transform.fit_transform(((x - mean) / scale).transpose(0, 2, 1)).astype(np.float64)
        scaler = StandardScaler().fit(features)
        model = Ridge(alpha=selected['alpha']).fit(scaler.transform(features), target, sample_weight=weights(y, groups))
        path = output / f'minirocket-{seed}.joblib'
        joblib.dump(dict(rocket=transform, scaler=scaler, model=model, mean=mean, scale=scale), path)
        members.append(dict(path=path.name, sha256=file_sha256(path), seed=seed))
    identity['versions'] = {p:version(p) for p in ['aeon', 'numpy', 'scikit-learn', 'numba']}
    return dict(identity=identity, candidates=reports, selected_alpha=selected['alpha'], members=members,
                readout_precision='float64 after MiniRocket transform',
                source_validation_brier=selected['mean_brier'], head='standardized MiniRocket features plus ridge one-hot regression; clipped/normalized class scores')


def neural_spec(method, channels, classes):
    if method == 'inceptiontime':
        path = Path('research/representation/learned_baselines/configs/analysis/inceptiontime-followup-260911.yaml')
        config = yaml.safe_load(path.read_text())
        spec = {**config['architecture'], 'input_channels': channels, 'output_classes': classes,
                'return_binary_difference': False}
        return spec, dict(maximum_epochs=1500, minimum_epochs=100, patience=100), [11, 23, 47, 71, 101]
    path = Path('research/representation/neurotycho/configs/analysis/neurotycho-transfer-260910.yaml')
    spec = yaml.safe_load(path.read_text())['raw_encoder']
    return {**spec, 'output_classes': classes}, dict(maximum_epochs=400, minimum_epochs=30, patience=30), [11, 23, 47]


def make_network(method, spec):
    if method == 'inceptiontime':
        from research.representation.learned_baselines.inceptiontime_baseline import InceptionNetwork
        return InceptionNetwork(spec)
    from research.representation.shared.representation_state_neural import make_encoder
    return make_encoder(spec)


def network_prediction(model, x, batch=16):
    import torch
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(x[i:i+batch]), dim=1).cpu().numpy() for i in range(0,len(x),batch)])


def fit_network(method, x, y, groups, spec, stopping, lr, seed, valid=None, epochs=None):
    import torch
    from research.representation.shared.representation_state_neural import seed_torch
    seed_torch(seed)
    model = make_network(method, spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=50, min_lr=1e-5)
    target = torch.as_tensor(y, dtype=torch.long)
    w = torch.as_tensor(weights(y, groups), dtype=torch.float32)
    best, stale, best_state, best_epoch, history = np.inf, 0, None, 0, []
    started = time.monotonic()
    for epoch in range(1, (epochs or stopping['maximum_epochs']) + 1):
        model.train(); loss_total = 0.
        for indices in torch.randperm(len(y)).split(16):
            optimizer.zero_grad()
            loss = (torch.nn.functional.cross_entropy(model(x[indices]), target[indices], reduction='none') * w[indices]).mean()
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite neural training loss')
            loss.backward(); optimizer.step(); loss_total += float(loss.detach()) * len(indices)
        scheduler.step(loss_total / len(y))
        if valid is not None:
            vx, vy, vg = valid
            score = metrics(vy, network_prediction(model, vx), vg)['brier']
            if score < best - 1e-8:
                best, best_epoch, stale = score, epoch, 0
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else: stale += 1
            if epoch % 25 == 0: history.append(dict(epoch=epoch, training_loss=loss_total/len(y), validation_brier=score))
            if epoch >= stopping['minimum_epochs'] and stale >= stopping['patience']: break
    if valid is not None: model.load_state_dict(best_state)
    else: best_epoch = epoch
    return model, dict(best_epoch=best_epoch, epochs_run=epoch, seconds=time.monotonic()-started,
        source_training_metrics=metrics(y, network_prediction(model, x), groups), history=history,
        reached_epoch_limit=epoch == (epochs or stopping['maximum_epochs']))


def neural(root, method, x, y, groups, identity, output):
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    spec, stopping, seeds = neural_spec(method, x.shape[-1], int(y.max())+1)
    identity.update(spec=spec, stopping=stopping, torch=version('torch'), numpy=version('numpy'))
    progress_path = output / f'{method}-progress.json'
    completed = {}
    if progress_path.exists():
        old = json.loads(progress_path.read_text())
        if old['identity'] != identity: raise ValueError('neural progress identity changed')
        completed = old['completed']
    reports = []
    for lr in [1e-3, 1e-4]:
        fold_reports = []
        for fi, (train, valid) in enumerate(folds(y, groups)):
            key = f'{lr}-{fi}'
            if key in completed:
                fold_reports.append(completed[key]); continue
            mean, scale = normalization(x[train])
            tx = torch.as_tensor((x[train]-mean)/scale)
            vx = torch.as_tensor((x[valid]-mean)/scale)
            model, report = fit_network(method, tx, y[train], groups[train], spec, stopping, lr, 11,
                                         valid=(vx, y[valid], groups[valid]))
            p = network_prediction(model, vx)
            report.update(training_rows=train.tolist(), validation_rows=valid.tolist(), probability=p.tolist(),
                          **metrics(y[valid], p, groups[valid]))
            completed[key] = report; fold_reports.append(report)
            atomic_json(progress_path, dict(identity=identity, completed=completed))
            print(method, lr, fi, report['brier'], report['best_epoch'], flush=True)
        reports.append(dict(lr=lr, folds=fold_reports, mean_brier=float(np.mean([f['brier'] for f in fold_reports]))))
    selected = min(reports, key=lambda r:r['mean_brier'])
    epochs = int(np.median([f['best_epoch'] for f in selected['folds']]))
    mean, scale = normalization(x); tensor = torch.as_tensor((x-mean)/scale)
    members = []
    for seed in seeds:
        path = output / f'{method}-{seed}.pt'
        member_key = f'final-{seed}'
        if member_key in completed:
            member = completed[member_key]
            if file_sha256(path) != member['sha256']:
                raise ValueError('resumed neural member checksum mismatch')
            members.append(member)
            continue
        model, report = fit_network(method, tensor, y, groups, spec, stopping, selected['lr'], seed, epochs=epochs)
        torch.save(dict(spec=spec, state_dict=model.state_dict(), mean=torch.as_tensor(mean), scale=torch.as_tensor(scale)), path)
        member = dict(path=path.name, sha256=file_sha256(path), seed=seed, training=report)
        members.append(member)
        completed[member_key] = member
        atomic_json(progress_path, dict(identity=identity, completed=completed))
        print('Saved', method, seed, epochs, flush=True)
    return dict(identity=identity, candidates=reports, selected=selected, members=members,
                source_validation_brier=selected['mean_brier'], final_epochs=epochs)


def screen(root, method, report, manifest, output):
    # Called only after all source decisions and model hashes have been written.
    x, y, groups, rows = load_role(root, manifest, 'screen')
    p = []
    for member in report['members']:
        path = output / member['path']
        if file_sha256(path) != member['sha256']: raise ValueError('saved model checksum mismatch')
        if method == 'minirocket':
            fitted = joblib.load(path)
            raw = ((x-fitted['mean'])/fitted['scale']).transpose(0,2,1)
            features = fitted['scaler'].transform(fitted['rocket'].transform(raw).astype(np.float64))
            p.append(probabilities(fitted['model'].predict(features)))
        else:
            import torch
            saved = torch.load(path, map_location='cpu', weights_only=True)
            model = make_network(method, saved['spec']);model.load_state_dict(saved['state_dict'])
            tensor = (torch.as_tensor(x)-saved['mean'])/saved['scale']
            p.append(network_prediction(model,tensor))
    probability = np.mean(p,axis=0)
    atomic_json(root/f'{method}-screen.json',dict(**metrics(y,probability,groups), probability=probability.tolist(),
        y=y.tolist(),group=groups.tolist(),row_id=[r['row_id'] for r in rows],
        source_report_sha256=file_sha256(output/f'{method}.json'), stage='Exploratory discovery, not confirmation'))


def main(root, method):
    output=root/'screen-models';output.mkdir(exist_ok=True)
    if (output/f'{method}.json').exists(): raise FileExistsError('completed source fit exists')
    manifest=input_manifest(root)
    x,y,groups,rows=load_role(root,manifest,'source')
    identity=dict(manifest_sha256=file_sha256(root/'manifest.json'),raw_sha256=file_sha256(root/'raw-views.npz'),
        source_rows=[r['row_id'] for r in rows],source_groups=sorted(set(groups)),target_data_used=False,
        runner_sha256=file_sha256(Path(__file__)), normalization='Source-only channel mean/SD across records and time; within-record amplitude is retained',
        pretraining=False,augmentation=False)
    identity['helper_sha256'] = {name: file_sha256(Path(name)) for name in [
        'research/representation/dataset_screen/common.py',
        'research/representation/shared/representation_state_neural.py',
        'research/representation/learned_baselines/inceptiontime_baseline.py']}
    report=rocket(root,x,y,groups,identity,output) if method=='minirocket' else neural(root,method,x,y,groups,identity,output)
    atomic_json(output/f'{method}.json',report)
    screen(root,method,report,manifest,output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--method',choices=['minirocket','inceptiontime','aligned_encoder'],required=True)
    args=parser.parse_args();main(args.root,args.method)
