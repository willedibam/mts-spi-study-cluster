"""Pack, source-fit and evaluate the bounded InceptionTime follow-up."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
import yaml

from src.inceptiontime_baseline import InceptionNetwork, fit_network
from src.neurotycho_learning import predict_binary
from src.representation_screen import training_subsets
from src.representation_state_data import source_pool_for_seed, observed_view

CONFIG = Path('configs/analysis/inceptiontime-followup-260911.yaml')
MODULES = [Path(__file__), Path('src/inceptiontime_baseline.py'), Path('src/neurotycho_learning.py'),
           Path('src/representation_state_neural.py')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def pack(args):
    if (args.root / 'manifest.json').exists():
        raise FileExistsError('pack already exists; verify/reuse it instead of overwriting')
    args.root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text())
    original = yaml.safe_load(Path('configs/analysis/oscillatory-coorganization-pilot-260909.yaml').read_text())
    cases, provenance, artifacts = [], {}, {}

    def save(name, bank):
        path = args.root / (name + '.npz')
        np.savez_compressed(path, **bank)
        artifacts[path.name] = sha(path)

    for name, folder in [('original', 'oscillatory_coorganization_pilot_260909'),
                         ('faster_dynamics', 'oscillatory_coorganization_transfer_260909'),
                         ('direct_mechanism', 'oscillatory_mechanism_transfer_260910')]:
        root = Path('data') / folder
        manifest = json.loads((root / 'manifest.json').read_text())
        for artifact, digest in manifest['artifacts'].items():
            assert sha(root / artifact) == digest, artifact
        provenance[name] = dict(manifest_sha256=sha(root / 'manifest.json'), artifacts=manifest['artifacts'])
        rows = manifest['rows']; masters = np.load(root / 'masters.npy', mmap_mode='r')
        def bank_for(ix):
            return dict(x=np.stack([observed_view(masters[rows[i]['master_index']], 16, 1000) for i in ix]).astype(np.float32),
                        y=np.asarray([rows[i]['target'] for i in ix]),
                        record_id=np.asarray([rows[i]['row_id'] for i in ix]),
                        master_id=np.asarray([rows[i]['master_id'] for i in ix]))
        evaluation = [i for i,r in enumerate(rows) if r['role'] == 'evaluation' and r['M'] == 16 and r['T'] == 1000]
        assert len(evaluation) == 200
        save('target-' + name, bank_for(evaluation))
        if name != 'original':
            continue
        pool = np.asarray([i for i,r in enumerate(rows) if r['role'] == 'training_pool'])
        assert len(pool) == 120
        source = bank_for(pool); save('source-synthetic', source)
        assert not set(source['master_id']) & {rows[i]['master_id'] for i in evaluation}
        strata = np.asarray([r['coupling_index'] for r in rows]); lookup = {int(row):i for i,row in enumerate(pool)}
        for cohort_seed in config['cohort_seeds']:
            cohort = source_pool_for_seed(rows, pool, original, cohort_seed)
            for n, train in training_subsets(strata, cohort, [b//2 for b in config['synthetic_label_budgets']], cohort_seed).items():
                indices = np.asarray([lookup[int(i)] for i in train])
                cv = StratifiedKFold(2, shuffle=True, random_state=cohort_seed)
                folds = [dict(fit=indices[a].tolist(), validation=indices[b].tolist())
                         for a,b in cv.split(train, strata[train])]
                cases.append(dict(name=f'synthetic-n{2*n}-c{cohort_seed}', domain='synthetic',
                    source='source-synthetic.npz', train=indices.tolist(), folds=folds,
                    original_train_indices=train.tolist(), labels=2*n, cohort_seed=cohort_seed))
    source_path = Path('data/neurotycho_neural_260910/matched.npz')
    source = load_npz(source_path)
    assert source['x'].shape == (352,2000,16)
    save('source-neurotycho', source)
    provenance['neurotycho_source'] = dict(path=str(source_path), sha256=sha(source_path))
    for animal in config['target_animals']:
        indices = np.flatnonzero(source['animal'] != animal)
        groups = sorted(set(source['animal'][indices])); assert len(groups) == 3
        folds = [dict(fit=indices[source['animal'][indices] != group].tolist(),
                      validation=indices[source['animal'][indices] == group].tolist(), animal=group) for group in groups]
        cases.append(dict(name=f'neurotycho-{animal}', domain='neurotycho', source='source-neurotycho.npz',
                          train=indices.tolist(), folds=folds, excluded_animal=animal, labels=len(indices)))
    target_path = Path('results/neurotycho_library_followup_260911/evaluation/bank.npz')
    target_meta = json.loads(target_path.with_suffix('.json').read_text())
    assert sha(target_path) == target_meta['sha256']
    with np.load(target_path) as data:
        target = {k:data[k] for k in ['x','y','record_id','animal','archive']}
    raw = target['x'].transpose(0,2,1)
    target['x'] = ((raw - raw.mean(1,keepdims=True)) / raw.std(1,keepdims=True)).astype(np.float32)
    assert target['x'].shape == (128,2000,16)
    save('target-neurotycho', target)
    provenance['neurotycho_target'] = dict(path=str(target_path), sha256=sha(target_path))
    write_json(args.root / 'manifest.json', dict(config_sha256=sha(CONFIG), cases=cases,
               artifacts=artifacts, provenance=provenance, target_inspected_before_followup=True))
    print('Packed', len(cases), 'cases', flush=True)


def fit_case(args):
    config = yaml.safe_load(CONFIG.read_text()); training = config['training']
    manifest = json.loads((args.root / 'manifest.json').read_text())
    assert manifest['config_sha256'] == sha(CONFIG)
    case = next(c for c in manifest['cases'] if c['name'] == args.case)
    path = args.root / case['source']; assert sha(path) == manifest['artifacts'][path.name]
    source = load_npz(path); train = np.asarray(case['train'])
    # Excluded animals and noncohort rows never become training tensors.
    if case['domain'] == 'neurotycho':
        assert case['excluded_animal'] not in source['animal'][train]
    local = {int(i):j for j,i in enumerate(train)}
    x = torch.from_numpy(source['x'][train]).to(args.device)
    y = source['y'][train]
    archives = source['archive'][train] if case['domain'] == 'neurotycho' else None
    folder = args.root / 'fits' / args.case; folder.mkdir(parents=True, exist_ok=True)
    identity = dict(case=case, config_sha256=sha(CONFIG), source_sha256=sha(path),
        modules={str(p):sha(p) for p in MODULES}, torch=torch.__version__, numpy=np.__version__,
        device=args.device, platform=platform.platform())
    if args.smoke:
        started = time.perf_counter()
        model, report = fit_network(x[:4], y[:4], None, config['architecture'], training, .001, 0, 1729, epochs=2)
        p = predict_binary(model, x[:4])
        assert p.shape == (4,) and np.isfinite(p).all()
        state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        replay = InceptionNetwork(config['architecture']); replay.load_state_dict(state)
        delta = float(np.abs(predict_binary(replay, x[:4].cpu())-p).max())
        assert delta < 1e-4
        write_json(folder/'smoke.json', dict(identity=identity, report=report, replay_delta=delta,
            seconds=time.perf_counter()-started, input_shape=list(x.shape)))
        print('SMOKE', args.case, report['seconds'], delta, flush=True); return
    for member_seed in config['member_seeds']:
        stem = folder / f'member-{member_seed}'
        if stem.with_suffix('.json').exists():
            report = json.loads(stem.with_suffix('.json').read_text())
            assert report['identity'] == identity and report['checkpoint_sha256'] == sha(stem.with_suffix('.pt'))
            assert report['predictions_sha256'] == sha(stem.with_suffix('.npz'))
            print('Verified existing', stem, flush=True); continue
        start = time.perf_counter(); choices = []
        for lr, decay in itertools.product(training['learning_rates'], training['weight_decays']):
            folds = []
            for fold, indices in enumerate(case['folds']):
                a = np.asarray([local[i] for i in indices['fit']]); b = np.asarray([local[i] for i in indices['validation']])
                assert not set(a)&set(b) and set(a)|set(b) == set(range(len(train)))
                model, log = fit_network(x[a], y[a], None if archives is None else archives[a],
                    config['architecture'], training, lr, decay, member_seed+1000*fold, validation=(x[b],y[b]))
                probability = predict_binary(model, x[b], training['batch_size'])
                folds.append(dict(**log, validation_probability=probability.tolist(),
                    validation_y=y[b].tolist(), training_ids=source['record_id'][train[a]].tolist(),
                    validation_ids=source['record_id'][train[b]].tolist()))
                print(dict(case=args.case, member=member_seed, lr=lr, decay=decay, fold=fold,
                    best_epoch=log['best_epoch'], brier=log['validation_brier'], seconds=log['seconds']), flush=True)
                del model
            choices.append(dict(lr=lr, decay=decay, folds=folds,
                                mean_brier=float(np.mean([f['validation_brier'] for f in folds]))))
            write_json(stem.with_suffix('.progress.json'), dict(identity=identity, candidates=choices))
        chosen = min(choices, key=lambda c:c['mean_brier'])
        epochs = max(1, int(np.median([f['best_epoch'] for f in chosen['folds']])))
        model, log = fit_network(x, y, archives, config['architecture'], training, chosen['lr'], chosen['decay'],
                                member_seed, epochs=epochs)
        torch.save(dict(spec=config['architecture'], state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),
                   stem.with_suffix('.pt'))
        np.savez_compressed(stem.with_suffix('.npz'), training_ids=source['record_id'][train], y=y,
                            probability=predict_binary(model,x,training['batch_size']))
        write_json(stem.with_suffix('.json'), dict(identity=identity, member_seed=member_seed,
            candidates=choices, selected_lr=chosen['lr'], selected_decay=chosen['decay'], selected_epochs=epochs,
            final=log, total_seconds=time.perf_counter()-start, target_data_used=False,
            checkpoint_sha256=sha(stem.with_suffix('.pt')), predictions_sha256=sha(stem.with_suffix('.npz'))))
        print('COMPLETED', stem, flush=True)


def metrics(y, p):
    return dict(balanced_accuracy=float(balanced_accuracy_score(y,p>=.5)),
                auroc=float(roc_auc_score(y,p)), brier=float(np.mean((y-p)**2)))


def evaluate(args):
    config = yaml.safe_load(CONFIG.read_text()); manifest = json.loads((args.root/'manifest.json').read_text())
    assert manifest['config_sha256'] == sha(CONFIG)
    # Global source-fit barrier before opening a target pack.
    frozen = {}
    for case in manifest['cases']:
        for seed in config['member_seeds']:
            stem = args.root/'fits'/case['name']/f'member-{seed}'
            report = json.loads(stem.with_suffix('.json').read_text())
            assert report['target_data_used'] is False and report['identity']['case'] == case
            assert report['identity']['config_sha256'] == sha(CONFIG)
            assert sha(stem.with_suffix('.pt')) == report['checkpoint_sha256']
            assert sha(stem.with_suffix('.npz')) == report['predictions_sha256']
            frozen[str(stem)] = dict(report_sha256=sha(stem.with_suffix('.json')), checkpoint_sha256=report['checkpoint_sha256'])
    output = args.root/'evaluation'
    if (output/'report.json').exists():
        raise FileExistsError('evaluation already complete; use independent verification')
    output.mkdir(exist_ok=True)
    summaries, audits = [], []
    for case in manifest['cases']:
        source = load_npz(args.root/case['source']); train = np.asarray(case['train'])
        datasets = config['evaluation']['synthetic'] if case['domain']=='synthetic' else ['neurotycho']
        for dataset in datasets:
            path = args.root/f'target-{dataset}.npz'
            assert sha(path) == manifest['artifacts'][path.name]
            target = load_npz(path)
            if dataset == 'neurotycho':
                ix = np.flatnonzero(target['animal']==case['excluded_animal'])
                target = {k:v[ix] for k,v in target.items()}
            else:
                assert not set(source['master_id'][train]) & set(target['master_id'])
            predictions = []
            for seed in config['member_seeds']:
                stem = args.root/'fits'/case['name']/f'member-{seed}'
                checkpoint = torch.load(stem.with_suffix('.pt'),map_location=args.device,weights_only=True)
                model = InceptionNetwork(checkpoint['spec']).to(args.device); model.load_state_dict(checkpoint['state_dict'])
                saved = load_npz(stem.with_suffix('.npz'))
                np.testing.assert_array_equal(saved['training_ids'],source['record_id'][train])
                np.testing.assert_array_equal(saved['y'],source['y'][train])
                replay = predict_binary(model,torch.from_numpy(source['x'][train]).to(args.device))
                delta = float(np.abs(replay-saved['probability']).max())
                assert delta < 1e-4
                p = predict_binary(model,torch.from_numpy(target['x']).to(args.device))
                predictions.append(p)
                audits.append(dict(case=case['name'],dataset=dataset,seed=seed,source_replay_delta=delta))
            probabilities = np.asarray(predictions)
            ensemble = probabilities.mean(0)
            np.savez_compressed(output/f'{case["name"]}-{dataset}.npz', probability=probabilities,
                ensemble=ensemble, y=target['y'], record_id=target['record_id'],
                archive=target.get('archive',np.full(len(ensemble),dataset)))
            groups = target.get('archive',np.full(len(ensemble),dataset))
            for member, p in [*zip(config['member_seeds'],probabilities),('ensemble',ensemble)]:
                dates = {str(g):metrics(target['y'][groups==g],p[groups==g]) for g in np.unique(groups)}
                score = {k:float(np.mean([d[k] for d in dates.values()])) for k in next(iter(dates.values()))}
                summaries.append(dict(case=case['name'],dataset=dataset,member=member,**score,groups=dates))
    write_json(output/'report.json',dict(config_sha256=sha(CONFIG),manifest_sha256=sha(args.root/'manifest.json'),
        frozen=frozen,scores=summaries,source_replays=audits,device=args.device,
        prediction_hashes={p.name:sha(p) for p in sorted(output.glob('*.npz'))}))
    print('Completed evaluation',output,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['pack','fit','evaluate'])
    parser.add_argument('--root',type=Path,default=Path('results/inceptiontime_followup_260911'))
    parser.add_argument('--case')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    {'pack':pack,'fit':fit_case,'evaluate':evaluate}[args.stage](args)
