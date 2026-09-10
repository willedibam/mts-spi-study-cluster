"""Independent local metric, label, coverage and source-selection audit."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, p):
    # Deliberately do not call the production reporting/metric implementation.
    negative, positive = p[y == 0], p[y == 1]
    pairs = positive[:, None] - negative[None, :]
    return dict(balanced_accuracy=float(((negative < .5).mean() + (positive >= .5).mean()) / 2),
                auroc=float(((pairs > 0) + .5 * (pairs == 0)).mean()),
                balanced_brier=float(((negative**2).mean() + ((positive-1)**2).mean()) / 2))


def main():
    target = Path('results/neurotycho_target_pilot_260910')
    root = target / 'evaluation'
    source = Path('results/neurotycho_source_pilot_260910')
    report = json.loads((root / 'report.json').read_text())
    assert sha(root / 'predictions.npz') == report['prediction_sha256']
    assert sha(Path('configs/analysis/neurotycho-transfer-260910.yaml')) == report['protocol_sha256']
    assert sha(source / 'cuda-replay-audit.json') == report['source_cuda_precision_audit_sha256']
    assert report['threshold'] == .5 and report['target_calibration'] is False
    assert report['source_model_selection_only'] is True and not report['excluded_target_dates']
    assert len(report['models']) == 32 and len(report['scores']) == 32
    labels = {}
    for path in target.glob('201*.json'):
        meta = json.loads(path.read_text())
        assert meta['usable']
        for r in meta['records']:
            assert r['quality']['accepted']
            rid = f'{r["archive"]}/{r["session"]}/{r["state"]}/{r["start"]}'
            assert rid not in labels
            labels[rid] = (r['target'], r['animal'], r['archive'])
    assert len(labels) == 128
    with np.load(root / 'predictions.npz') as data:
        bank = {k: data[k] for k in data.files}
    assert len(bank['y']) == 4096
    keys = list(zip(bank['method'], bank['seed'], bank['record_id'], bank['M'], bank['T']))
    assert len(set(keys)) == 4096
    assert np.isfinite(bank['probability']).all()
    assert np.all((bank['probability'] >= 0) & (bank['probability'] <= 1))
    for rid, y, a, date in zip(bank['record_id'], bank['y'], bank['animal'], bank['archive']):
        assert labels[rid] == (y, a, date)
    maximum, details = 0., []
    metric_errors = {k: 0. for k in ['balanced_accuracy', 'auroc', 'balanced_brier']}
    for score in report['scores']:
        ix = (bank['method'] == score['method']) & (bank['seed'] == score['seed']) & (bank['M'] == score['M'])
        assert set(bank['record_id'][ix]) == set(labels) and ix.sum() == 128
        assert np.all(bank['T'][ix] == score['T'])
        date_values = {}
        for date in score['dates']:
            jx = ix & (bank['archive'] == date['archive'])
            assert jx.sum() == 32 and np.sum(bank['y'][jx] == 1) == 16
            computed = metrics(bank['y'][jx], bank['probability'][jx])
            date_values[date['archive']] = computed
            maximum = max(maximum, max(abs(computed[k]-date[k]) for k in computed))
            for k in computed:
                metric_errors[k] = max(metric_errors[k], abs(computed[k]-date[k]))
            details.append(dict(method=score['method'], seed=score['seed'], M=score['M'],
                animal=date['animal'], date=date['archive'][:8], **computed,
                false_positives=int(np.sum((bank['y'][jx] == 0) & (bank['probability'][jx] >= .5))),
                false_negatives=int(np.sum((bank['y'][jx] == 1) & (bank['probability'][jx] < .5)))))
        animals = []
        for a in score['animals']:
            rows = [date_values[r['archive']] for r in score['dates'] if r['animal'] == a['animal']]
            means = {k: np.mean([r[k] for r in rows]) for k in computed}
            maximum = max(maximum, max(abs(means[k]-a[k]) for k in means))
            animals.append(means)
        maximum = max(maximum, max(abs(np.mean([a[k] for a in animals])-score['mean'][k]) for k in computed))
    # Production neural Brier arithmetic was float32; saved probabilities
    # are promoted to float64 here. BA and rank-based AUC must still agree.
    assert metric_errors['balanced_accuracy'] < 1e-12 and metric_errors['auroc'] < 1e-12
    assert maximum < 1e-7
    # Confirm local copies correspond to every source report used by evaluation.
    for model in report['models']:
        name = Path(model['model']).name
        folder = ('source-enriched' if name.startswith('enriched-') else 'source-neural'
                  if name.startswith('matched-') else 'source-pooling' if name.startswith('pool-')
                  else 'source-statistical')
        assert sha(source / folder / name) == model['report_sha256']
    with np.load('data/neurotycho_neural_260910/dense.npz') as data:
        dense = dict(zip(data['record_id'], data['animal']))
    with np.load('data/neurotycho_neural_260910/matched.npz') as data:
        matched = dict(zip(data['record_id'], data['animal']))
    source_rows, cv_error = [], 0.
    for path in sorted((source / 'source-enriched').glob('*.json')):
        fit = json.loads(path.read_text())
        assert sha(path.with_suffix('.pt')) == fit['checkpoint_sha256']
        assert sha(path.with_suffix('.npz')) == fit['predictions_sha256']
        excluded = fit['identity']['animal']
        for candidate in fit['candidates']:
            values = []
            for fold in candidate['folds']:
                training = set(fold['training_ids']); validation = set(fold['validation_ids'])
                assert all(dense[r] not in [excluded, fold['animal']] for r in training)
                assert {dense[r] for r in training} == set(fit['source_animals']) - {fold['animal']}
                assert all(matched[r] == fold['animal'] for r in validation)
                assert not training.intersection(validation)
                value = metrics(np.asarray(fold['validation_y']), np.asarray(fold['validation_probability']))['balanced_brier']
                cv_error = max(cv_error, abs(value-fold['best_validation_brier']))
                values.append(value)
            cv_error = max(cv_error, abs(np.mean(values)-candidate['mean_brier']))
        selected = min(fit['candidates'], key=lambda c: c['mean_brier'])
        assert (selected['lr'], selected['decay']) == (fit['selected_lr'], fit['selected_decay'])
        assert int(np.median([f['best_epoch'] for f in selected['folds']])) == fit['selected_epochs']
        source_rows.append(dict(model=path.name, selected_source_cv_brier=selected['mean_brier'],
            selected_epochs=fit['selected_epochs'], training_windows=fit['training_windows'], seconds=fit['total_seconds']))
    assert len(source_rows) == 6 and cv_error < 1e-7
    result = dict(status='passed', report_sha256=sha(root/'report.json'), prediction_rows=4096,
        unique_target_windows=128, models=32, score_groups=32, dates=4, animals=2,
        metric_max_difference=maximum, per_date_metric_errors=metric_errors, enriched_source_cv_max_difference=cv_error,
        enriched_source_models=source_rows, per_date_details=details,
        scope='Independent label/coverage/metric/source-selection audit; full checkpoint CPU replay and PF Gram verification recorded in original report.')
    (root / 'independent-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'per_date_details'}, indent=2))


if __name__ == '__main__':
    main()
