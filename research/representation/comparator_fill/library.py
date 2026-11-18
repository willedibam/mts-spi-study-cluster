"""Time-series library comparators on raw windows: ROCKET, MiniRocket, MultiRocket-Hydra and QUANT (aeon 1.5).

Runs in the separate aeon environment. Each dataset keeps its own split and metric:
- discovery: fit on source rows, score the screen rows (and Siena, unchanged fits, post hoc);
- fmri: fit on source rows, score each held-out role as participant means;
- cohort: the participant-grouped five-fold CV (one repeat by default) or NeuroTycho's leave-target-animal-out transfer.

One head for every method and dataset, so the methods differ only in their features: features are standardized
on the training rows and read by ridge regression on one-hot targets, the penalty chosen on grouped folds of the
training rows (the transform is label-free and is fitted once on those training rows). QUANT keeps its published
extremely-randomised-trees head. This differs from the stored MiniRocket rows of fMRI and NeuroTycho (logistic
head), so MiniRocket is re-run here under the common head as the reference.
"""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from research.representation.dataset_screen.common import folds, metrics, probabilities, weights

ALPHAS = [1e-3, 1e-2, 1e-1, 1., 10., 100., 1e3, 1e4, 1e5, 1e6]
METHODS = ['rocket', 'minirocket', 'multirocket_hydra', 'quant']


class Features:
    """A label-free transform of (n, T, M) windows, fitted on the training rows."""
    def __init__(self, name, seed, jobs):
        from aeon.transformations.collection import convolution_based as c
        self.name = name
        if name == 'rocket':
            self.parts = [c.Rocket(n_kernels=10000, n_jobs=jobs, random_state=seed)]
        elif name == 'minirocket':
            self.parts = [c.MiniRocket(n_kernels=10000, max_dilations_per_kernel=32, n_jobs=jobs, random_state=seed)]
        elif name == 'multirocket_hydra':
            self.parts = [c.MultiRocket(n_jobs=jobs, random_state=seed), c.HydraTransformer(n_jobs=jobs, random_state=seed)]
        else:
            from aeon.transformations.collection.interval_based import QUANTTransformer
            self.parts = [QUANTTransformer()]

    def fit(self, x):
        for p in self.parts:
            p.fit(np.ascontiguousarray(x.transpose(0, 2, 1), dtype=np.float32))
        return self

    def transform(self, x):
        raw = np.ascontiguousarray(x.transpose(0, 2, 1), dtype=np.float32)
        return np.hstack([np.asarray(p.transform(raw), dtype=np.float64) for p in self.parts])


def inner_folds(y, groups, fold_groups):
    if fold_groups is not None:
        return fold_groups(groups)
    return folds(y, groups)


def fit_predict(name, x, y, groups, train, tests, seeds, jobs, weighted=True, fold_groups=None):
    """Class probabilities for each index set in tests, averaged over seeds; also the chosen penalties."""
    classes = int(y.max()) + 1
    target = np.eye(classes)[y[train]]
    w = weights(y[train], groups[train]) if weighted else None
    outputs, chosen = [np.zeros((len(t), classes)) for t in tests], []
    for seed in seeds:
        transform = Features(name, seed, jobs).fit(x[train])
        f_train = transform.transform(x[train])
        f_tests = [transform.transform(x[t]) for t in tests]
        if name == 'quant':
            model = ExtraTreesClassifier(200, max_features=.1, criterion='entropy', n_jobs=jobs, random_state=seed)
            model.fit(f_train, y[train], sample_weight=w)
            for out, f in zip(outputs, f_tests):
                out += model.predict_proba(f) / len(seeds)
            continue
        scaler = StandardScaler().fit(f_train)
        f_train = scaler.transform(f_train)
        loss = np.zeros(len(ALPHAS))
        for a, b in inner_folds(y[train], groups[train], fold_groups):
            wa = None if w is None else weights(y[train][a], groups[train][a])
            for k, alpha in enumerate(ALPHAS):
                p = probabilities(Ridge(alpha=alpha).fit(f_train[a], target[a], sample_weight=wa).predict(f_train[b]))
                loss[k] += metrics(y[train][b], p, groups[train][b])['brier']
        alpha = ALPHAS[int(np.argmin(loss))]
        chosen.append(alpha)
        model = Ridge(alpha=alpha).fit(f_train, target, sample_weight=w)
        for out, f in zip(outputs, f_tests):
            out += probabilities(model.predict(scaler.transform(f))) / len(seeds)
    return outputs, chosen


def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class='ovr', average='macro'))


def discovery(root, siena, methods, jobs):
    from research.representation.dataset_screen.raw import input_manifest, load_role, normalization
    manifest = input_manifest(root)
    xs, ys, gs, _ = load_role(root, manifest, 'source')
    xt, yt, gt, rows = load_role(root, manifest, 'screen')
    mean, scale = normalization(xs)
    parts, sizes = [xs, xt], [len(xs), len(xt)]
    if siena is not None:
        xc, yc, gc, crow = load_role(siena, input_manifest(siena), 'confirmation')
        parts.append(xc); sizes.append(len(xc))
    x = (np.concatenate(parts) - mean) / scale
    edges = np.cumsum([0] + sizes)
    train = np.arange(edges[1])
    tests = [np.arange(edges[k], edges[k + 1]) for k in range(1, len(sizes))]
    y = np.concatenate([ys, yt] + ([yc] if siena is not None else []))
    g = np.concatenate([gs, gt] + ([gc] if siena is not None else []))
    report = dict(screen={}, siena={}, row_id=[r['row_id'] for r in rows])
    for name in methods:
        p, chosen = fit_predict(name, x, y, g, train, tests, [11, 23, 47], jobs)
        report['screen'][name] = dict(**metrics(yt, p[0], gt), auroc_pooled=auroc(yt, p[0]), probability=p[0].tolist(), alpha=chosen)
        if siena is not None:
            report['siena'][name] = dict(**metrics(yc, p[1], gc), auroc_pooled=auroc(yc, p[1]), probability=p[1].tolist())
            report['siena_row_id'] = [r['row_id'] for r in crow]
        print(root.name, name, {k: v for k, v in report['screen'][name].items() if k != 'probability'}, flush=True)
    return report


def fmri(root, methods, jobs):
    from research.representation.empirical_transfer.evaluate_fmri import ROLES, group_metrics
    from research.representation.empirical_transfer.splits import source_folds
    with np.load(root / 'bank.npz', allow_pickle=False) as archive:
        x, y, g, role, rows = (archive[k] for k in ('x', 'y', 'group', 'role', 'row_id'))
    x = np.asarray(x, dtype=np.float64)
    train = np.flatnonzero(role == 'source')
    tests = [np.flatnonzero(role == r) for r in ROLES]
    report = {r: {} for r in ROLES}
    for name in methods:
        p, chosen = fit_predict(name, x, y, g, train, tests, [11, 23, 47], jobs, fold_groups=source_folds)
        for r, ix, q in zip(ROLES, tests, p):
            per = group_metrics(y[ix], q[:, 1], g[ix])
            report[r][name] = dict({m: float(np.mean([v[m] for v in per.values()])) for m in ('brier', 'balanced_accuracy', 'auroc')},
                                   auroc_pooled=auroc(y[ix], q), probability=q[:, 1].tolist(), row_id=rows[ix].tolist(), alpha=chosen)
        print('fmri', name, {k: v for k, v in report['confirmation_primary'][name].items() if k not in ('probability', 'row_id')}, flush=True)
    return report


def cohort(name_cohort, root, methods, jobs, repeats):
    from research.representation.comparator_fill.cohort import CONTRASTS, aggregate, splits, summary, windows
    bank = np.load(root / 'bank.npz')
    x = windows(root, bank)
    y_all, groups_all = bank['y'], bank['group'].astype(str)
    roles_all = bank['role'].astype(str) if 'role' in bank.files else np.full(len(y_all), '')
    animals_all = groups_all
    if name_cohort == 'neurotycho':
        groups_all = np.array([f'{g}-{r}' for g, r in zip(groups_all, roles_all)])
    report = {}
    for label, positive, negative in CONTRASTS[name_cohort]:
        ix = np.flatnonzero(np.isin(y_all, [positive, negative]))
        y, groups, roles, animals = (y_all[ix] == positive).astype(int), groups_all[ix], roles_all[ix], animals_all[ix]
        fit_groups = animals if name_cohort == 'neurotycho' else groups
        outer = splits(name_cohort, y, groups, roles, animals, repeats)
        used = max(r for r, _, _ in outer) + 1
        entry = dict(scores={}, units={})
        for name in methods:
            window_p = np.full((used, len(y)), np.nan)
            for r, train, test in outer:
                p, _ = fit_predict(name, x[ix], y, fit_groups, train, [test], [11], jobs, weighted=False)
                window_p[r, test] = p[0][:, 1]
            keys, unit_label, unit_p = aggregate(name_cohort, y, groups, animals, window_p, used)
            entry['scores'][name] = summary(unit_label, unit_p)
            entry['units'][name] = unit_p.tolist()
            print(name_cohort, label, name, entry['scores'][name], flush=True)
        entry.update(unit_keys=keys.tolist(), unit_labels=unit_label.tolist(), repeats=used)
        report[label] = entry
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol', choices=['discovery', 'fmri', 'cohort'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--siena', type=Path)
    parser.add_argument('--cohort')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', nargs='*', default=METHODS)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=1)
    a = parser.parse_args()
    if a.protocol == 'discovery':
        result = discovery(a.root, a.siena, a.methods, a.jobs)
    elif a.protocol == 'fmri':
        result = fmri(a.root, a.methods, a.jobs)
    else:
        result = cohort(a.cohort, a.root, a.methods, a.jobs, a.repeats)
    result['versions'] = {p: version(p) for p in ('aeon', 'numpy', 'scikit-learn', 'numba')}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=1) + '\n')
