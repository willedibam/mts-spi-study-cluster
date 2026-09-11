"""Conventional feature baselines for the declared full-size follow-up."""
import hashlib
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def standardized(x):
    """N,M,T -> standardized float32 N,M,T, using population SD."""
    x = np.asarray(x, dtype=np.float64)
    return ((x-x.mean(-1, keepdims=True))/x.std(-1, keepdims=True)).astype(np.float32)


def channel_layout(features, layout):
    if layout == 'channels':
        return features.reshape(len(features), -1)
    if layout != 'pooled':
        raise ValueError(layout)
    x = np.where(np.isfinite(features), features, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.concatenate([np.nanmean(x, axis=1), np.nanstd(x, axis=1),
            *[np.nanquantile(x, q, axis=1) for q in [.25, .5, .75]]], axis=1)


class FiniteColumns(BaseEstimator, TransformerMixin):
    def __init__(self, fraction=.95, minimum_sd=1e-8):
        self.fraction = fraction
        self.minimum_sd = minimum_sd

    def fit(self, x, y=None):
        self.finite_ = np.isfinite(x).mean(0) >= self.fraction
        clean = np.where(np.isfinite(x[:, self.finite_]), x[:, self.finite_], np.nan)
        self.imputer_ = SimpleImputer(strategy='median')
        clean = self.imputer_.fit_transform(clean)
        self.variable_ = clean.std(0) > self.minimum_sd
        if not self.variable_.any():
            raise ValueError('No nonconstant finite training features')
        return self

    def transform(self, x):
        clean = np.where(np.isfinite(x[:, self.finite_]), x[:, self.finite_], np.nan)
        return self.imputer_.transform(clean)[:, self.variable_]


def fit_head(x, y, archive, c, config):
    weights = np.empty(len(y))
    for a in np.unique(archive):
        for label in [0, 1]:
            ix = (archive == a) & (y == label)
            assert ix.any()
            weights[ix] = 1/ix.sum()
    weights *= len(y)/weights.sum()
    model = make_pipeline(FiniteColumns(config['minimum_finite_fraction'], config['variance_threshold']),
        StandardScaler(), LogisticRegression(C=c, max_iter=config['logistic_max_iter'], solver='lbfgs'))
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y, logisticregression__sample_weight=weights)
    return model


def brier(y, p):
    return float(np.mean([np.mean((p[y == label]-label)**2) for label in [0, 1]]))


def make_rocket(seed, config, threads):
    from aeon.transformations.collection.convolution_based import MiniRocket
    return MiniRocket(n_kernels=config['minirocket_kernels'],
        max_dilations_per_kernel=config['minirocket_max_dilations'], random_state=seed, n_jobs=threads)


def predict_model(model, bank):
    if model['method'] == 'minirocket':
        x = model['rocket'].transform(standardized(bank['x']))
    else:
        x = channel_layout(bank[model['method']], model['layout'])
    return model['head'].predict_proba(x)[:, 1]


def fit_model(bank, method, seed, config, threads=2):
    groups = sorted(set(bank['animal']))
    assert len(groups) == 3
    layouts = ['multivariate'] if method == 'minirocket' else config['feature_layouts']
    candidates = [dict(layout=layout, C=c, folds=[]) for layout in layouts for c in config['logistic_C']]
    raw = standardized(bank['x']) if method == 'minirocket' else None
    for group in groups:
        train = np.flatnonzero(bank['animal'] != group)
        valid = np.flatnonzero(bank['animal'] == group)
        rocket = None
        if method == 'minirocket':
            rocket = make_rocket(seed, config, threads)
            tx = rocket.fit_transform(raw[train])
            vx = rocket.transform(raw[valid])
        for layout in layouts:
            if method != 'minirocket':
                tx = channel_layout(bank[method][train], layout)
                vx = channel_layout(bank[method][valid], layout)
            for candidate in [c for c in candidates if c['layout'] == layout]:
                head = fit_head(tx, bank['y'][train], bank['archive'][train], candidate['C'], config)
                p = head.predict_proba(vx)[:, 1]
                candidate['folds'].append(dict(animal=group, training_ids=bank['record_id'][train].tolist(),
                    validation_ids=bank['record_id'][valid].tolist(), y=bank['y'][valid].tolist(),
                    probability=p.tolist(), brier=brier(bank['y'][valid], p)))
        print(f'{method} seed{seed}: source fold {group} complete', flush=True)
    for c in candidates:
        c['mean_brier'] = float(np.mean([f['brier'] for f in c['folds']]))
    selected = min(candidates, key=lambda c: c['mean_brier'])
    rocket = None
    if method == 'minirocket':
        rocket = make_rocket(seed, config, threads)
        x = rocket.fit_transform(raw)
    else:
        x = channel_layout(bank[method], selected['layout'])
    head = fit_head(x, bank['y'], bank['archive'], selected['C'], config)
    return dict(method=method, seed=seed, layout=selected['layout'], rocket=rocket, head=head), dict(
        source_animals=groups, candidates=candidates, selected={k:selected[k] for k in ['layout', 'C', 'mean_brier']},
        training_ids=bank['record_id'].tolist(), input_width=x.shape[1])
