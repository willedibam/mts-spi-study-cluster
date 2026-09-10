"""Source-animal-only selection for the frozen NeuroTycho comparison."""
import itertools

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.interaction_share_learning import fit_statistical, ridge_readout
from src.neurotycho_learning import balanced_brier, date_state_weights
from src.representation_screen import fit_view


METHODS = {'m-pca': ('m', 'pca'), 'mg-pca': ('m+g', 'pca'),
           'z-pca': ('z', 'pca'), 'z-pls': ('z', 'pls'),
           'mz-pca': ('m+z', 'pca'), 'validity-pca': ('validity', 'pca'),
           'spectrum': ('spectral', 'logistic')}


def spectral_view(values, relative):
    return values.reshape(len(values), 5, 14)[:, :, 6:].reshape(len(values), 40) if relative else values


def spectrum_fit(bank, indices, choice):
    x = spectral_view(bank['spectral'][indices], choice['relative'])
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          LogisticRegression(C=choice['C'], max_iter=2000, solver='lbfgs'))
    weights = date_state_weights(bank['archive'][indices], bank['y'][indices])
    model.fit(x, bank['y'][indices], logisticregression__sample_weight=weights)
    return model


def predict_fitted(fitted, bank, indices):
    if fitted['method'] == 'spectrum':
        x = spectral_view(bank['spectral'][indices], fitted['choice']['relative'])
        return fitted['model'].predict_proba(x)[:, 1]
    x = fitted['transform'].transform(bank, indices)
    return np.clip(fitted['model'].predict(x).reshape(-1), 0, 1)


def fit_grouped(bank, method, config):
    """Accept only the three source animals after the target has been discarded."""
    if method not in METHODS or len(set(bank['animal'])) != 3:
        raise ValueError('expected a frozen method and three source animals')
    if not np.all(bank['M'] == 16) or not np.all(bank['T'] == 2000):
        raise ValueError('fitting requires full source observations')
    groups = sorted(set(bank['animal']))
    folds = [(np.flatnonzero(bank['animal'] != group), np.flatnonzero(bank['animal'] == group))
             for group in groups]
    view, head = METHODS[method]
    if head == 'logistic':
        # Prefer the smaller spectral view and stronger regularization on ties.
        choices = [dict(relative=r, C=c) for r, c in itertools.product([True, False], sorted(config['spectral_C_grid']))]
    elif head == 'pca':
        choices = [dict(components=k, alpha=a) for k, a in itertools.product(
            sorted(config['pca_caps']), sorted(config['ridge_alpha_grid'], reverse=True))]
    else:
        choices = [dict(components=k) for k in sorted(config['pls_components'])]
    reports = [dict(choice=choice, folds=[]) for choice in choices]
    for group, (train, valid) in zip(groups, folds, strict=True):
        cached = {}
        for report in reports:
            choice = report['choice']
            if head == 'logistic':
                fitted = dict(method=method, choice=choice, model=spectrum_fit(bank, train, choice))
                prediction = predict_fitted(fitted, bank, valid)
            elif head == 'pca':
                k = choice['components']
                if k not in cached:
                    transform, x = fit_view(bank, view, train, {**config['preprocessing'], 'pca_dimensions': k})
                    cached[k] = x, transform.transform(bank, valid)
                x, vx = cached[k]
                model = ridge_readout(x, bank['y'][train], head, choice['alpha'])
                prediction = np.clip(model.predict(vx).reshape(-1), 0, 1)
            else:
                transform, model = fit_statistical(bank, view, train, bank['y'], config['preprocessing'],
                                                   head, choice['components'])
                prediction = np.clip(model.predict(transform.transform(bank, valid)).reshape(-1), 0, 1)
            if not np.isfinite(prediction).all():
                raise FloatingPointError('nonfinite source-validation prediction')
            report['folds'].append(dict(animal=str(group),
                training_ids=bank['record_id'][train].tolist(), validation_ids=bank['record_id'][valid].tolist(),
                y=bank['y'][valid].tolist(), probability=prediction.tolist(),
                balanced_brier=balanced_brier(bank['y'][valid], prediction)))
    for report in reports:
        report['mean_brier'] = float(np.mean([fold['balanced_brier'] for fold in report['folds']]))
    selected = min(reports, key=lambda r: r['mean_brier'])
    choice = selected['choice']
    all_rows = np.arange(len(bank['y']))
    if head == 'logistic':
        fitted = dict(method=method, choice=choice, model=spectrum_fit(bank, all_rows, choice))
    else:
        transform, model = fit_statistical(bank, view, all_rows, bank['y'], config['preprocessing'],
                                           head, choice['components'], choice.get('alpha'))
        fitted = dict(method=method, choice=choice, transform=transform, model=model)
    return fitted, dict(source_animals=groups, candidates=reports, selected=choice,
                        selected_source_validation_brier=selected['mean_brier'])
