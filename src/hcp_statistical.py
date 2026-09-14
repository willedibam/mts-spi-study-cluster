"""Source-only HCP fixed-feature comparators, using the declared prior grids."""
import itertools

import numpy as np

from src.hcp_grouping import family_folds, participant_balanced_brier, participant_state_weights
from src.interaction_share_learning import fit_statistical, ridge_readout
from src.neurotycho_library_baselines import fit_head, make_rocket, standardized
from src.representation_screen import fit_view


VIEWS = {'m-pca': ('m', 'pca'), 'z-pca': ('z', 'pca'), 'mz-pca': ('m+z', 'pca'),
         'm-pls': ('m', 'pls'), 'z-pls': ('z', 'pls')}


def predict(fitted, bank):
    if fitted['method'] == 'minirocket':
        x = fitted['rocket'].transform(standardized(bank['x'].transpose(0, 2, 1)))
        return fitted['model'].predict_proba(x)[:, 1]
    if fitted['method'] == 'spectra':
        return fitted['model'].predict_proba(bank['spectra'])[:, 1]
    x = fitted['transform'].transform(bank, np.arange(len(bank['y'])))
    return np.clip(fitted['model'].predict(x).reshape(-1), 0, 1)


def fit_source(bank, method, config, fold_seed, *, member_seed=11, threads=2):
    """The bank must contain source records only; no confirmation input is accepted."""
    y = np.asarray(bank['y'])
    people, families = np.asarray(bank['participant']), np.asarray(bank['family'])
    ids = np.asarray(bank['record_id'])
    if ids.shape != y.shape or len(np.unique(ids)) != len(y):
        raise ValueError('Expected unique source record identifiers')
    participant_state_weights(y, people)
    folds = family_folds(people, families, config['selection']['folds'], fold_seed)
    if method in VIEWS:
        view, head = VIEWS[method]
        choices = ([dict(components=k, alpha=a) for k, a in itertools.product(
            sorted(config['pca_caps']), sorted(config['ridge_alpha_grid'], reverse=True))]
                   if head == 'pca' else [dict(components=k) for k in sorted(config['pls_components'])])
    elif method in {'spectra', 'minirocket'}:
        choices = [dict(C=c) for c in sorted(config['linear_classification']['logistic_C'])]
    else:
        raise ValueError(f'Unknown declared HCP comparator: {method}')
    reports = [dict(choice=choice, folds=[], oof=np.full(len(y), np.nan)) for choice in choices]
    raw = standardized(bank['x'].transpose(0, 2, 1)) if method == 'minirocket' else None
    for train, valid in folds:
        cache = {}
        if method == 'minirocket':
            rocket = make_rocket(member_seed, config['minirocket'], threads)
            tx, vx = rocket.fit_transform(raw[train]), rocket.transform(raw[valid])
        elif method == 'spectra':
            tx, vx = bank['spectra'][train], bank['spectra'][valid]
        for report in reports:
            choice = report['choice']
            if method in {'spectra', 'minirocket'}:
                model = fit_head(tx, y[train], people[train], choice['C'], config['linear_classification'])
                probability = model.predict_proba(vx)[:, 1]
            elif head == 'pca':
                k = choice['components']
                if k not in cache:
                    transform, tx = fit_view(bank, view, train, {**config['preprocessing'], 'pca_dimensions': k})
                    cache[k] = tx, transform.transform(bank, valid)
                tx, vx = cache[k]
                model = ridge_readout(tx, y[train], head, choice['alpha'])
                probability = np.clip(model.predict(vx).reshape(-1), 0, 1)
            else:
                transform, model = fit_statistical(bank, view, train, y, config['preprocessing'], head, choice['components'])
                probability = np.clip(model.predict(transform.transform(bank, valid)).reshape(-1), 0, 1)
            score = participant_balanced_brier(y[valid], probability, people[valid])
            report['oof'][valid] = probability
            report['folds'].append(dict(training_ids=ids[train].tolist(), validation_ids=ids[valid].tolist(),
                                        probability=probability.tolist(), participant_brier=score))
    for report in reports:
        report['mean_participant_brier'] = participant_balanced_brier(y, report['oof'], people)
        report['oof_probability'] = report.pop('oof').tolist()
    selected = min(reports, key=lambda item: item['mean_participant_brier'])
    choice = selected['choice']
    fitted = dict(method=method, choice=choice)
    if method == 'minirocket':
        fitted['rocket'] = make_rocket(member_seed, config['minirocket'], threads)
        tx = fitted['rocket'].fit_transform(raw)
    elif method == 'spectra':
        tx = bank['spectra']
    if method in {'spectra', 'minirocket'}:
        fitted['model'] = fit_head(tx, y, people, choice['C'], config['linear_classification'])
    else:
        fitted['transform'], fitted['model'] = fit_statistical(bank, view, np.arange(len(y)), y,
            config['preprocessing'], head, choice['components'], choice.get('alpha'))
    return fitted, dict(candidates=reports, selected=choice, member_seed=member_seed,
                         record_ids=ids.tolist(), y=y.tolist(), participants=people.tolist(),
                         family_count=len(np.unique(families)), participant_count=len(np.unique(people)),
                         fold_seed=fold_seed, target_data_used=False)
