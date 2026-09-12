"""Bounded native MiniRocket/ridge comparison on frozen full-size synthetic data."""
import hashlib
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from aeon.classification.convolution_based import MiniRocketClassifier
import joblib
import numpy as np
from scipy.stats import rankdata

ROOT = Path('results/minirocket_synthetic_followup_260912')
DATA = Path('results/inceptiontime_followup_260911')
PROTOCOL = Path('docs/minirocket-synthetic-followup.md')
SEEDS = [11, 23, 47]
TARGETS = ['original', 'faster_dynamics', 'direct_mechanism']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def load(path):
    with np.load(path) as data:
        bank = {key: data[key] for key in data.files}
    # Frozen neural packs are N,T,M; aeon requires N,M,T.
    assert bank['x'].ndim == 3 and bank['x'].shape[1:] == (1000, 16)
    bank['x'] = np.ascontiguousarray(bank['x'].transpose(0, 2, 1))
    return bank


def metric(y, score):
    positive = y == 1
    negative = ~positive
    n1, n0 = positive.sum(), negative.sum()
    return dict(balanced_accuracy=float(((score[positive] > 0).mean() + (score[negative] <= 0).mean()) / 2),
                auroc=float((rankdata(score)[positive].sum() - n1*(n1+1)/2) / (n1*n0)))


def main(evaluate_only=False):
    ROOT.mkdir(exist_ok=True)
    fits = ROOT / 'fits'; fits.mkdir(exist_ok=True)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    cases = [case for case in manifest['cases'] if case['domain'] == 'synthetic']
    assert len(cases) == 9
    source_path = DATA / 'source-synthetic.npz'
    assert sha(source_path) == manifest['artifacts'][source_path.name]
    source = load(source_path)
    assert source['x'].shape == (120, 16, 1000)
    versions = {name: importlib.metadata.version(name) for name in ['aeon', 'scikit-learn', 'numpy', 'numba']}
    assert versions['aeon'] == '1.5.0'
    identity = dict(script_sha256=sha(Path(__file__)), protocol_sha256=sha(PROTOCOL),
                    manifest_sha256=sha(DATA / 'manifest.json'), source_sha256=sha(source_path), versions=versions)
    if (ROOT / 'identity.json').exists():
        original = json.loads((ROOT / 'identity.json').read_text())
        if evaluate_only:
            assert {k:v for k,v in original.items() if k != 'script_sha256'} == {k:v for k,v in identity.items() if k != 'script_sha256'}
            assert all((fits / f'{case["name"]}-s{seed}.json').exists() for case in cases for seed in SEEDS)
            identity = original  # Preserve original source code identity; no refits.
        else:
            assert original == identity
    else:
        write(ROOT / 'identity.json', identity)
    for case in cases:
        ix = np.asarray(case['train'])
        assert len(ix) == case['labels'] and len(np.unique(ix)) == len(ix)
        for seed in SEEDS:
            stem = fits / f'{case["name"]}-s{seed}'
            if stem.with_suffix('.json').exists():
                saved = json.loads(stem.with_suffix('.json').read_text())
                assert saved['identity'] == identity and saved['case'] == case
                assert sha(stem.with_suffix('.joblib')) == saved['model_sha256']
                continue
            assert not evaluate_only
            started = time.perf_counter()
            model = MiniRocketClassifier(n_kernels=10000, max_dilations_per_kernel=32, n_jobs=2, random_state=seed)
            model.fit(source['x'][ix], source['y'][ix])
            assert model.classes_.tolist() == [0, 1]
            score = model.pipeline_.decision_function(source['x'][ix])
            np.testing.assert_array_equal(model.predict(source['x'][ix]), (score > 0).astype(int))
            joblib.dump(model, stem.with_suffix('.joblib'))
            restored = joblib.load(stem.with_suffix('.joblib'))
            np.testing.assert_allclose(restored.pipeline_.decision_function(source['x'][ix]), score, atol=1e-12, rtol=0)
            np.savez_compressed(stem.with_suffix('.npz'), record_id=source['record_id'][ix], y=source['y'][ix], score=score)
            write(stem.with_suffix('.json'), dict(identity=identity, case=case, seed=seed,
                  target_data_used=False, training_ids=source['record_id'][ix].tolist(),
                  selected_alpha=float(model._estimator.alpha_), feature_count=int(model._scaler.n_features_in_),
                  model_sha256=sha(stem.with_suffix('.joblib')), predictions_sha256=sha(stem.with_suffix('.npz')),
                  seconds=time.perf_counter()-started))
            print('FITTED', case['name'], seed, float(model._estimator.alpha_), flush=True)
    # All 27 source models must be frozen before any target is opened.
    frozen = {}
    for case in cases:
        for seed in SEEDS:
            stem = fits / f'{case["name"]}-s{seed}'
            saved = json.loads(stem.with_suffix('.json').read_text())
            assert saved['identity'] == identity and saved['case'] == case and not saved['target_data_used']
            assert sha(stem.with_suffix('.joblib')) == saved['model_sha256']
            assert sha(stem.with_suffix('.npz')) == saved['predictions_sha256']
            frozen[stem.name] = sha(stem.with_suffix('.json'))
    output = ROOT / 'evaluation'; output.mkdir(exist_ok=True)
    if (output / 'report.json').exists():
        raise FileExistsError('Evaluation exists; independently verify instead of repeating.')
    rows, replays = [], []
    for dataset in TARGETS:
        path = DATA / f'target-{dataset}.npz'
        assert sha(path) == manifest['artifacts'][path.name]
        target = load(path)
        assert not set(source['master_id']) & set(target['master_id'])
        assert target['x'].shape == (200, 16, 1000)
        for case in cases:
            for seed in SEEDS:
                stem = fits / f'{case["name"]}-s{seed}'
                model = joblib.load(stem.with_suffix('.joblib'))
                features = model._transformer.transform(target['x'])
                scaled = model._scaler.transform(features)
                score = model._estimator.decision_function(scaled)
                labels = model._estimator.predict(scaled)
                np.testing.assert_array_equal(labels, (score > 0).astype(int))
                scores = metric(target['y'], score)
                # Separate implementation of the reported metrics.
                from sklearn.metrics import balanced_accuracy_score, roc_auc_score
                assert abs(scores['balanced_accuracy'] - balanced_accuracy_score(target['y'], labels)) < 1e-12
                assert abs(scores['auroc'] - roc_auc_score(target['y'], score)) < 1e-12
                name = f'{stem.name}-{dataset}.npz'
                if (output / name).exists():
                    with np.load(output / name) as old:
                        np.testing.assert_array_equal(old['score'], score)
                np.savez_compressed(output / name, record_id=target['record_id'], y=target['y'], score=score, prediction=labels)
                sample = np.r_[np.flatnonzero(target['y'] == 0)[:4], np.flatnonzero(target['y'] == 1)[:4]]
                np.testing.assert_array_equal(model._transformer.transform(target['x'][sample]), features[sample])
                score64 = (scaled.astype(np.float64) @ model._estimator.coef_.astype(np.float64).T + model._estimator.intercept_).ravel()
                assert np.max(np.abs(score64 - score)) < 1e-5
                delta = float(np.abs(model.pipeline_.decision_function(target['x'][sample]) - score[sample]).max())
                assert delta < 1e-5  # Float32 BLAS batch-size variation; features above remain exact.
                replays.append(delta)
                rows.append(dict(case=case['name'], labels=case['labels'], cohort=case['cohort_seed'],
                                 seed=seed, dataset=dataset, prediction_file=name, **scores))
        print('EVALUATED', dataset, flush=True)
    summaries = [dict(dataset=dataset, labels=n, **{metric: float(np.mean([row[metric] for row in rows if row['dataset'] == dataset and row['labels'] == n])) for metric in ['balanced_accuracy', 'auroc']})
                 for dataset in TARGETS for n in [10, 20, 40]]
    assert len(rows) == 81 and len(frozen) == 27
    write(output / 'report.json', dict(identity=identity, evaluation_script_sha256=sha(Path(__file__)), frozen=frozen, scores=rows, summary=summaries,
          prediction_hashes={p.name: sha(p) for p in output.glob('*.npz')},
          sampled_prediction_replays=len(replays), maximum_replay_error=max(replays),
          aggregation='Mean metrics over three source cohorts and three transform seeds; no ensemble',
          scores_are_probabilities=False))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-only', action='store_true', help='Require all original source fits; allow evaluator-only changes')
    main(parser.parse_args().evaluate_only)
