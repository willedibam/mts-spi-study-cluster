"""Named character axes on existing audited banks: prediction and readable effects.

Every model is fitted on the bank's source rows only and scored on its held-out
role (CHB-MIT source models are also applied unchanged to Siena). Each feature
block is imputed, scaled and, above 32 columns, PCA-compressed inside the
source fit, then blocks are concatenated for fusion.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mtsspi.character_axes import AXES, axis_scores, edge_profiles, level_scores
from research.representation.dataset_screen.common import folds, metrics, probabilities

TEST_ROLE = dict(lanl='screen', chbmit='screen', hmc='screen', grabmyo='screen',
                 neurotycho='target', fmri='confirmation_primary')
PREDICTIVE = [a.name for a in AXES if a.role == 'axis']


def readout(bank):
    spi = [str(s) for s in bank['spi_order']]
    M = int(bank['M'][0])
    if not np.all(bank['M'] == M):
        raise ValueError('axis readout needs one montage size per bank')
    profiles = edge_profiles(bank['aligned'], M, len(spi))
    organisation, pair_values, pairs, missing = axis_scores(profiles, spi)
    levels, level_names = level_scores(profiles, spi)
    names = [a.name for a in AXES]
    keep = [names.index(n) for n in PREDICTIVE]
    pair_keep = [k for k, p in enumerate(pairs) if p[0] in PREDICTIVE]
    return dict(axes=organisation[:, keep], levels=levels, pairs=pair_values[:, pair_keep],
                controls=organisation[:, [names.index('identity'), names.index('estimator')]],
                pair_names=[pairs[k] for k in pair_keep], level_names=level_names, missing=missing)


def block(width):
    steps = [SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()]
    return steps + ([PCA(32, random_state=261031)] if width > 32 else [])


class Blocks:
    def __init__(self, names): self.names = names
    def fit(self, X, y=None):
        self.pipes = [make_pipeline(*block(X[n].shape[1])).fit(X[n]) for n in self.names]
        return self
    def transform(self, X):
        return np.hstack([p.transform(X[n]) for p, n in zip(self.pipes, self.names)])


def fit(X, y, groups, names):
    best = None
    for C in (.01, .1, 1, 10):
        p = np.zeros((len(y), len(np.unique(y))))
        for train, valid in folds(y, groups):
            sub = {k: v[train] for k, v in X.items()}
            b = Blocks(names).fit(sub)
            model = LogisticRegression(C=C, max_iter=5000).fit(b.transform(sub), y[train])
            p[valid] = model.predict_proba(b.transform({k: v[valid] for k, v in X.items()}))
        score = metrics(y, probabilities(p), groups)['brier']
        if best is None or score < best[0]: best = (score, C)
    b = Blocks(names).fit(X)
    model = LogisticRegression(C=best[1], max_iter=5000).fit(b.transform(X), y)
    return lambda Z: model.predict_proba(b.transform(Z)), best


def score(y, p, groups):
    result = metrics(y, probabilities(p), groups)
    if p.shape[1] == 2:
        result['auroc'] = float(roc_auc_score(y, p[:, 1]))
    return result


def effects(values, y, groups, roles, test_role, names):
    """Class contrast of group means, separately on source and held-out groups."""
    out = {}
    if len(np.unique(y)) != 2:
        return out
    for k, name in enumerate(names):
        row = {}
        for role in ('source', test_role):
            ix = roles == role
            g = np.unique(groups[ix])
            means = np.array([np.nanmean(values[ix & (groups == u), k]) for u in g])
            lab = np.array([y[ix & (groups == u)][0] for u in g])
            a, c = means[lab == 1], means[lab == 0]
            if len(a) < 2 or len(c) < 2 or len(np.unique(y[ix & (groups == g[0])])) > 1:
                # Groups contain both classes (within-person designs): use paired window means.
                a = np.array([np.nanmean(values[ix & (groups == u) & (y == 1), k]) for u in g])
                c = np.array([np.nanmean(values[ix & (groups == u) & (y == 0), k]) for u in g])
                d = a - c
                row[role] = dict(mean_difference=float(np.nanmean(d)), paired_sd=float(np.nanstd(d, ddof=1)) if len(d) > 1 else None,
                                 groups_positive=int(np.sum(d > 0)), groups=len(d))
            else:
                pooled = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(c, ddof=1)) / 2)
                row[role] = dict(mean_difference=float(np.nanmean(a) - np.nanmean(c)),
                                 standardized=float((np.nanmean(a) - np.nanmean(c)) / pooled) if pooled > 0 else None,
                                 groups=[len(a), len(c)])
        out[name] = row
    return out


def main(banks, output):
    sets = dict(spectral=['spectral'], pearson=['pearson'], z=['z'], axes=['axes'], axes_levels=['axes', 'levels'],
                axis_pairs=['pairs'], pearson_axes=['pearson', 'axes'], pearson_pairs=['pearson', 'pairs'],
                spectral_axes=['spectral', 'axes'], spectral_pearson=['spectral', 'pearson'],
                spectral_pearson_axes=['spectral', 'pearson', 'axes'])
    report = {}
    cache = {}
    for name in ['lanl', 'chbmit', 'hmc', 'grabmyo', 'neurotycho', 'fmri']:
        bank = np.load(banks / f'{name}.npz')
        r = readout(bank)
        X = dict(spectral=bank['spectral'], pearson=bank['pearson'], z=bank['z'], axes=r['axes'], levels=r['levels'], pairs=r['pairs'])
        y, groups, roles = bank['y'], bank['group'].astype(str), bank['role'].astype(str)
        src, test = roles == 'source', roles == TEST_ROLE[name]
        targets = {TEST_ROLE[name]: (X, y, groups, test)}
        if name == 'chbmit':
            s = np.load(banks / 'siena.npz'); rs = readout(s)
            if [str(v) for v in s['spi_order']] != [str(v) for v in bank['spi_order']]:
                raise ValueError('Siena catalogue differs')
            XS = dict(spectral=s['spectral'], pearson=s['pearson'], z=s['z'], axes=rs['axes'], levels=rs['levels'], pairs=rs['pairs'])
            targets['siena'] = (XS, s['y'], s['group'].astype(str), np.ones(len(s['y']), bool))
        entry = dict(records=int(len(y)), source=int(src.sum()),
                     control_axes=dict(identity_min=float(np.nanmin(r['controls'][:, 0])), estimator_mean=float(np.nanmean(r['controls'][:, 1]))),
                     missing_pairs=len(r['missing']), axis_finite_fraction=dict(zip(PREDICTIVE, np.isfinite(r['axes']).mean(0).round(3).tolist())),
                     scores={}, effects=effects(np.hstack([r['axes'], r['levels']]), y, groups, roles, TEST_ROLE[name],
                                                PREDICTIVE + [f'level_{n}' for n in r['level_names']]))
        for label, names in sets.items():
            Xs = {k: v[src] for k, v in X.items()}
            predict, best = fit(Xs, y[src], groups[src], names)
            entry['scores'][label] = dict(source_brier=best[0], C=best[1])
            for t, (XT, yt, gt, ix) in targets.items():
                entry['scores'][label][t] = score(yt[ix], predict({k: v[ix] for k, v in XT.items()}), gt[ix])
            print(name, label, {t: {k: round(v, 3) for k, v in entry['scores'][label][t].items()} for t in targets}, flush=True)
        report[name] = entry
        cache[name] = dict(axes=r['axes'], levels=r['levels'], pair_names=r['pair_names'])
    output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--banks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.banks, args.output)
