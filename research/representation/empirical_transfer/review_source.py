"""Audit source reports and nominate a small z panel before target evaluation.

Panel selection is exploratory on source data. It is not another unbiased
cross-validation score, and named statistics are not physiological mechanisms.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.linear_model import Ridge

from mtsspi.provenance import atomic_json, file_sha256
from mtsspi.representation_screen import fit_view
from research.representation.empirical_transfer.fit_source import SPEC
from research.representation.empirical_transfer.splits import source_folds, group_weights
from research.representation.shared.interaction_share_learning import ridge_readout


def source_bank(root, excluded=None):
    manifest = json.loads((root / 'manifest.json').read_text())
    digest = file_sha256(root / 'bank.npz')
    if digest != json.loads((root / 'bank.json').read_text())['sha256']:
        raise ValueError('feature bank checksum mismatch')
    with np.load(root / 'bank.npz', allow_pickle=False) as raw:
        keep = raw['role'] == 'source'
        if excluded:
            keep &= raw['group'] != excluded
        bank = {key: raw[key][keep] for key in ['z', 'y', 'group', 'row_id']}
    lookup = {row['row_id']: row for row in manifest['rows']}
    bank['date'] = np.array([lookup[row].get('archive', row) for row in bank['row_id']])
    return bank, digest


def selected_folds(report):
    if 'selected_C' in report:
        return report['candidates'][str(report['selected_C'])]
    if 'identity' in report:
        return report['selected']['folds']
    return next(c['folds'] for c in report['candidates'] if c['choice'] == report['selected'])


def review_reports(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    lookup = {r['row_id']: r for r in manifest['rows']}
    source_ids = [r['row_id'] for r in manifest['rows'] if r['role'] == 'source']
    summaries = {}
    for path in sorted((root / 'source-models').glob('*.json')):
        if path.name.endswith('-progress.json'):
            continue
        report = json.loads(path.read_text())
        excluded = report.get('excluded_animal')
        eligible = {r for r in source_ids if lookup[r]['group'] != excluded}
        raw = 'identity' in report
        if raw:
            if report['identity']['target_data_used'] or set(report['identity']['source_rows']) != eligible:
                raise ValueError('raw report used an unexpected cohort')
            for member in report['members']:
                if file_sha256(path.parent / member['path']) != member['sha256']:
                    raise ValueError('raw member checksum mismatch')
        elif report['target_data_used'] or file_sha256(path.with_suffix('.joblib')) != report['model_sha256']:
            raise ValueError('statistical model identity mismatch')
        seen, points = set(), []
        for fold in selected_folds(report):
            if raw:
                ids = report['identity']['source_rows']
                train = [ids[i] for i in fold['training_rows']]
                valid = [ids[i] for i in fold['validation_rows']]
            else:
                train, valid = fold['training_ids'], fold['validation_ids']
            if not set(train + valid) <= eligible or set(valid) & seen:
                raise ValueError('source folds overlap or contain forbidden rows')
            if set(train) != eligible - set(valid) or len(set(valid)) != len(valid):
                raise ValueError('source fold is not a complete disjoint partition')
            if {lookup[r]['group'] for r in train} & {lookup[r]['group'] for r in valid}:
                raise ValueError('participant/animal split leakage')
            if 'y' in fold and fold['y'] != [lookup[r]['target'] for r in valid]:
                raise ValueError('validation labels differ from the input manifest')
            seen.update(valid)
            for row, probability in zip(valid, fold['probability'], strict=True):
                if not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError('invalid probability')
                points.append((lookup[row]['group'], lookup[row]['target'], probability))
        if seen != eligible:
            raise ValueError('source validation does not cover the declared cohort')
        by_group = []
        for group in sorted({p[0] for p in points}):
            y = np.array([p[1] for p in points if p[0] == group])
            p = np.array([p[2] for p in points if p[0] == group])
            if set(y) != {0, 1}:
                raise ValueError('missing source state')
            by_group.append(dict(group=group,
                brier=float(np.mean([np.mean((p[y == c] - c) ** 2) for c in (0, 1)])),
                balanced_accuracy=float(np.mean([np.mean((p[y == c] >= .5) == c) for c in (0, 1)]))))
        if raw:
            selection_brier = float(np.mean([f['participant_brier'] for f in selected_folds(report)]))
        else:
            selection_brier = report['selected_source_validation_brier']
        summaries[path.stem] = dict(selection_brier=selection_brier,
            group_mean_brier=float(np.mean([r['brier'] for r in by_group])),
            group_mean_balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in by_group])),
            groups=by_group, report_sha256=file_sha256(path),
            scope='Selected source-CV predictions, optimistic after hyperparameter selection; not confirmation')
    return summaries


def coordinate_effects(values, bank):
    """Equal people/animals; NeuroTycho first averages state contrasts over dates."""
    effects = []
    for group in sorted(set(bank['group'])):
        g = bank['group'] == group
        contrasts = []
        for date in sorted(set(bank['date'][g])):
            ix = g & (bank['date'] == date)
            if set(bank['y'][ix]) == {0, 1}:
                contrasts.append(values[ix & (bank['y'] == 1)].mean(0) - values[ix & (bank['y'] == 0)].mean(0))
        if not contrasts:  # fMRI row IDs are dates only as placeholders.
            contrasts = [values[g & (bank['y'] == 1)].mean(0) - values[g & (bank['y'] == 0)].mean(0)]
        effects.append(np.mean(contrasts, axis=0))
    return np.array(effects)


def coefficient_map(transform, model, width):
    if len(transform.blocks) != 1 or transform.blocks[0].name != 'z' or transform.pca is None:
        raise ValueError('expected a nondegenerate linear z-PCA model')
    beta = transform.pca.components_.T @ np.asarray(model.coef_).reshape(-1)
    result = np.zeros(width)
    result[transform.blocks[0].keep] = beta / transform.blocks[0].scale
    return beta, result


def nominate_panel(root, excluded=None):
    bank, digest = source_bank(root, excluded)
    name = 'z-pca' + (f'-{excluded}' if excluded else '')
    model_path = root / 'source-models' / f'{name}.joblib'
    report = json.loads(model_path.with_suffix('.json').read_text())
    if file_sha256(model_path) != report['model_sha256'] or digest != report['bank_sha256']:
        raise ValueError('fitted z model provenance mismatch')
    fitted = joblib.load(model_path)
    transform, model = fitted['transform'], fitted['model']
    choice = report['selected']
    config = yaml.safe_load(SPEC.read_text())
    block = transform.blocks[0]
    values = block.transform(bank['z'])
    beta, full_beta = coefficient_map(transform, model, bank['z'].shape[1])
    effects = coordinate_effects(values, bank)
    contributions = effects.mean(0) * beta
    full_score = model.predict(transform.transform(bank, np.arange(len(bank['y'])))).reshape(-1)
    groups = sorted(set(bank['group']))
    # Whole-group omissions probe nomination stability with the already selected
    # hyperparameters; they are not nested estimates of generalization accuracy.
    folds = ([(np.flatnonzero(bank['group'] != g), np.flatnonzero(bank['group'] == g)) for g in groups]
             if excluded else source_folds(bank['group']))
    fold_beta = []
    for train, _ in folds:
        tx, x = fit_view(bank, 'z', train, {**config['preprocessing'], 'pca_dimensions': choice['components']})
        head = ridge_readout(x, bank['y'][train], 'pca', choice['alpha'])
        if not excluded:
            head.fit(x, bank['y'][train], sample_weight=group_weights(bank['group'][train], bank['y'][train]))
        fold_beta.append(coefficient_map(tx, head, bank['z'].shape[1])[1])
    signs = np.mean(np.sign(np.array(fold_beta)[:, block.keep]) == np.sign(full_beta[block.keep]), axis=0)
    direction = np.mean(np.sign(effects) == np.sign(effects.mean(0)), axis=0)
    candidates = np.flatnonzero((contributions > 0) & (signs == 1) & (direction >= .75))
    ranked = candidates[np.argsort(-contributions[candidates], kind='stable')]
    selected = []
    for ix in ranked:
        if all(abs(np.corrcoef(values[:, ix], values[:, old])[0, 1]) < .95 for old in selected):
            selected.append(int(ix))
        if len(selected) == 3:
            break
    # Schema is a list of signed, ordered-edge SPI-pair descriptors.
    features = json.loads((root / 'feature-schema.json').read_text())
    if not isinstance(features, list) or len(features) != bank['z'].shape[1]:
        raise ValueError('feature schema does not match the bank width')
    panel = []
    for ix in selected:
        feature = features[int(block.keep[ix])]
        panel.append(dict(index=int(block.keep[ix]), feature=feature,
            coefficient_sign_consistency=float(signs[ix]), group_effect_sign_consistency=float(direction[ix]),
            mean_state_difference=float(effects[:, ix].mean() * block.scale[ix]),
            contribution_to_mean_model_contrast=float(contributions[ix])))
    fidelity = None
    if selected:
        x = values[:, selected]
        x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)
        surrogate = Ridge(alpha=1.).fit(x, full_score, sample_weight=group_weights(bank['group'], bank['y']))
        fidelity = float(surrogate.score(x, full_score, sample_weight=group_weights(bank['group'], bank['y'])))
    return dict(source_groups=groups, source_rows=bank['row_id'].tolist(), excluded=excluded,
        bank_sha256=digest, model_sha256=file_sha256(model_path), panel=panel,
        mean_model_state_contrast=float(coordinate_effects(full_score[:, None], bank).mean()),
        panel_additive_contrast=float(sum(contributions[i] for i in selected)),
        descriptive_source_score_R2=fidelity, target_data_used=False,
        rule='Up to three positive additive contributors; coefficient sign agrees across three source refits; state-effect sign agrees in at least 75% of source groups; absolute source correlation below .95 between chosen coordinates.',
        limitation='Exploratory source nomination with fixed selected hyperparameters; R2 is descriptive training fidelity, not validation accuracy. Small contributions do not establish a compact explanation. Physical interpretation awaits estimator and waveform checks.')


def main(root):
    output = root / 'source-review.json'
    if output.exists():
        raise FileExistsError(output)
    excluded = ['Chibi', 'George'] if root.name.startswith('neurotycho') else [None]
    result = dict(reports=review_reports(root), panels={str(a): nominate_panel(root, a) for a in excluded},
                  runner_sha256=file_sha256(Path(__file__)), target_outcomes_evaluated=False)
    atomic_json(output, result)
    print(output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
