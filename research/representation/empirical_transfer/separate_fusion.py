"""Preserve the selected SPI projection and append source-fitted PCA(z, cap=50).

This bounded sensitivity was specified before inspecting confirmation outcomes.
It tests whether joint PCA hid complementarity by displacing baseline directions.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from mtsspi.provenance import atomic_json, file_sha256
from mtsspi.representation_screen import fit_view
from research.representation.empirical_transfer.fit_single_spi import source_inputs
from research.representation.empirical_transfer.fit_source import SPEC, SeparateProjection
from research.representation.empirical_transfer.splits import source_folds, group_weights, group_brier
from research.representation.shared.interaction_share_learning import ridge_readout


def projections(bank, train, preprocessing, base_components, z_cap):
    baseline, b = fit_view(bank, 'u', train, {**preprocessing, 'pca_dimensions': base_components})
    z, zx = fit_view(bank, 'z', train, {**preprocessing, 'pca_dimensions': z_cap})
    return SeparateProjection(baseline, z), np.concatenate([b, zx], axis=1)


def fit_source(bank, config, base_components, z_cap=50):
    reports = [dict(choice=dict(alpha=a, base_components=base_components, z_cap=z_cap), folds=[])
               for a in sorted(config['ridge_alpha_grid'], reverse=True)]
    for train, valid in source_folds(bank['group']):
        transform, x = projections(bank, train, config['preprocessing'], base_components, z_cap)
        vx = transform.transform(bank, valid)
        for report in reports:
            model = ridge_readout(x, bank['y'][train], 'pca', report['choice']['alpha'])
            model.fit(x, bank['y'][train], sample_weight=group_weights(bank['group'][train], bank['y'][train]))
            p = np.clip(model.predict(vx).reshape(-1), 0, 1)
            report['folds'].append(dict(training_ids=bank['row_id'][train].tolist(),
                validation_ids=bank['row_id'][valid].tolist(), y=bank['y'][valid].tolist(), probability=p.tolist(),
                person_balanced_brier=group_brier(bank['y'][valid], p, bank['group'][valid]),
                base_dimensions=transform.baseline.pca.n_components_, z_dimensions=transform.z.pca.n_components_))
    for report in reports:
        report['mean_brier'] = float(np.mean([f['person_balanced_brier'] for f in report['folds']]))
    selected = min(reports, key=lambda r: r['mean_brier'])
    transform, x = projections(bank, np.arange(len(bank['y'])), config['preprocessing'], base_components, z_cap)
    model = ridge_readout(x, bank['y'], 'pca', selected['choice']['alpha'])
    model.fit(x, bank['y'], sample_weight=group_weights(bank['group'], bank['y']))
    return dict(transform=transform, model=model), dict(selected=selected['choice'],
        selected_source_validation_brier=selected['mean_brier'], candidates=reports,
        final_base_dimensions=transform.baseline.pca.n_components_, final_z_dimensions=transform.z.pca.n_components_)


def main(root):
    output = root / 'source-models/separate-spi-z50.joblib'
    if output.exists():
        raise FileExistsError(output)
    bank, order, edges, digest = source_inputs(root)
    baseline_report = root / 'source-models/single-spi.json'
    baseline_info = json.loads(baseline_report.read_text())
    baseline_path = baseline_report.with_suffix('.joblib')
    if digest != baseline_info['bank_sha256'] or file_sha256(baseline_path) != baseline_info['model_sha256']:
        raise ValueError('baseline identity changed')
    index = baseline_info['spi_index']
    bank['u'] = bank['aligned'][:, index * edges:(index + 1) * edges]
    config = yaml.safe_load(SPEC.read_text())
    fitted, report = fit_source(bank, config, baseline_info['selected']['components'])
    baseline = joblib.load(baseline_path)
    rows = np.arange(len(bank['y']))
    np.testing.assert_allclose(fitted['transform'].baseline.transform(bank, rows),
                               baseline['transform'].transform(bank, rows), rtol=1e-10, atol=1e-10)
    fitted.update(spi_index=index, spi_name=order[index], edges_per_spi=edges)
    joblib.dump(fitted, output)
    report.update(source_rows=bank['row_id'].tolist(), source_groups=sorted(set(bank['group'])),
        spi_name=order[index], bank_sha256=digest, model_sha256=file_sha256(output),
        baseline_report_sha256=file_sha256(baseline_report), target_data_used=False,
        baseline_source_projection_verified_identical=True,
        source_runner_sha256=file_sha256(Path(__file__)),
        source_fit_completed_at=datetime.now(timezone.utc).isoformat(),
        scope='Separate baseline and z projections, no joint PCA; five ridge choices on source folds. This is a declared fusion-order/cap-50 sensitivity, not a replacement primary model.')
    atomic_json(output.with_suffix('.json'), report)
    print('Source fit frozen', report['selected_source_validation_brier'], report['selected'], flush=True)

    # Do not access held-out features, labels or earlier predictions until the
    # model and source decisions above have been saved and checksummed.
    from research.representation.empirical_transfer.evaluate_fmri import checked_bank, group_metrics, ROLES
    held = checked_bank(root)
    held['u'] = held['aligned'][:, index * edges:(index + 1) * edges]
    prediction = np.clip(fitted['model'].predict(fitted['transform'].transform(held, np.arange(len(held['y'])))), 0, 1)
    saved = json.loads((root / 'predictions-statistical.json').read_text())
    if saved['row_id'] != held['row_id'].tolist() or saved['lock_sha256'] != file_sha256(root / 'evaluation-lock.json'):
        raise ValueError('baseline prediction identity mismatch')
    base_prediction = np.array(saved['probability']['single-spi'])
    results = {}
    for role in ROLES:
        ix = held['role'] == role
        people = sorted(set(held['group'][ix]))
        draws = np.random.default_rng(261031).integers(0, len(people), (5000, len(people)))
        a = group_metrics(held['y'][ix], prediction[ix], held['group'][ix])
        b = group_metrics(held['y'][ix], base_prediction[ix], held['group'][ix])
        metrics = {}
        for metric in ['brier', 'balanced_accuracy', 'auroc']:
            av = np.array([a[g][metric] for g in people])
            bv = np.array([b[g][metric] for g in people])
            delta = (av - bv) * (-1 if metric == 'brier' else 1)
            metrics[metric] = dict(baseline=float(bv.mean()), separate_fusion=float(av.mean()),
                improvement=float(delta.mean()), conditional_95_interval=np.quantile(delta[draws].mean(1), [.025, .975]).tolist())
        results[role] = dict(participants=len(people), recordings=int(ix.sum()), metrics=metrics)
    atomic_json(root / 'separate-fusion-summary.json', dict(roles=results,
        source_report_sha256=file_sha256(output.with_suffix('.json')),
        primary_lock_sha256=file_sha256(root / 'evaluation-lock.json'),
        row_id=held['row_id'].tolist(), probability=prediction.tolist(),
        limitation='Secondary sensitivity specified before inspecting confirmation outcomes. Intervals condition on fitted models; no target refitting or threshold calibration.'))
    print('Frozen separate-projection predictions complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    main(parser.parse_args().root)
