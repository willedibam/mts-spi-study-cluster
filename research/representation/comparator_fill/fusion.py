"""X ⊕ z, z | X and commonality on the cohort banks, under the cohort protocol of cohort.py.

For each base block set X (Pearson r; Pearson + relative spectra; mean-SPI) and each contrast:
- X ⊕ z, stacked: X and z each keep their own fitted model (cohort.fit_predict: own PCA cap, own C). Their
  inner out-of-fold logits on the outer training rows train a two-weight logistic meta-learner, which then
  combines the two models' outer test logits. The meta-weights are reported.
- z | X: z's PCs regressed on X's PCs on the outer training rows; the residuals (that fit applied to the test
  rows) are what z carries that X cannot express linearly, scored with the same logistic protocol.
- Commonality on unit-level cross-validated log-loss (bits): U(z; X) = D(X) - D(X ⊕ z),
  U(X; z) = D(z) - D(X ⊕ z), shared C = D(null) - D(X ⊕ z) - U(z; X) - U(X; z). Negative C is suppression:
  the union exceeds what the two blocks give separately.
Nothing is selected on outer test rows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LinearRegression, LogisticRegression

from research.representation.comparator_fill import cohort as C
from research.representation.dataset_screen.common import folds

BASES = dict(r=['pearson'], rs=['pearson', 'spectral'], mean_spi=['mean_spi'])
EPS = 1e-6


def clipped_logit(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def oof(X, y, groups, names, train):
    """Inner out-of-fold probabilities on the outer training rows (each inner fit chooses its own C)."""
    p = np.full(len(train), np.nan)
    for a, b in folds(y[train], groups[train]):
        p[b] = C.fit_predict(X, y, groups, names, train[a], train[b])[0]
    return p


def stacked(X, y, groups, base, train, test):
    """Test probabilities of X ⊕ z and the meta-learner's [w_X, w_z]."""
    inner = np.column_stack([clipped_logit(oof(X, y, groups, n, train)) for n in (base, ['z'])])
    meta = LogisticRegression(C=1e4, max_iter=5000).fit(inner, y[train])
    outer = np.column_stack([clipped_logit(C.fit_predict(X, y, groups, n, train, test)[0]) for n in (base, ['z'])])
    return meta.predict_proba(outer)[:, 1], meta.coef_[0].tolist()


def residual_z(X, base, train):
    """z's PCs minus their linear prediction from X's PCs, the regression fitted on the training rows only."""
    bx = C.Blocks(base).fit({n: X[n][train] for n in base})
    bz = C.Blocks(['z']).fit({'z': X['z'][train]})
    fx = bx.transform({n: X[n] for n in base})
    fz = bz.transform({'z': X['z']})
    fit = LinearRegression().fit(fx[train], fz[train])
    return fz - fit.predict(fx)


def log_loss_bits(label, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(label * np.log2(p) + (1 - label) * np.log2(1 - p)))


def bootstrap_auroc(label, p, keys, n=2000, seed=261102):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    people = np.array([k.split('|')[0] for k in keys])
    unique = np.unique(people)
    values = []
    for _ in range(n):
        pick = np.concatenate([np.flatnonzero(people == u) for u in rng.choice(unique, len(unique))])
        if len(np.unique(label[pick])) == 2:
            values.append(roc_auc_score(label[pick], p[pick]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main(cohort, root, output, repeats):
    bank, X, x, _ = C.load(root)
    y_all, groups_all = bank['y'], bank['group'].astype(str)
    roles_all = bank['role'].astype(str) if 'role' in bank.files else np.full(len(y_all), '')
    animals_all = groups_all
    if cohort == 'neurotycho':
        groups_all = np.array([f'{g}-{r}' for g, r in zip(groups_all, roles_all)])
    bases = {k: v for k, v in BASES.items() if all(b in X for b in v)}
    report = dict(cohort=cohort, scope=__doc__.splitlines()[0], contrasts={})
    for label, positive, negative in C.CONTRASTS[cohort]:
        ix = np.flatnonzero(np.isin(y_all, [positive, negative]))
        y, groups, roles, animals = (y_all[ix] == positive).astype(int), groups_all[ix], roles_all[ix], animals_all[ix]
        Xc = {k: v[ix] for k, v in X.items()}
        fit_groups = animals if cohort == 'neurotycho' else groups
        outer = C.splits(cohort, y, groups, roles, animals, repeats)
        used = max(r for r, _, _ in outer) + 1
        rows = ['z'] + [f'{k}{s}' for k in bases for s in ('', '+z', '|z')]
        window_p = {n: np.full((used, len(y)), np.nan) for n in rows}
        weights = {k: [] for k in bases}
        prior = np.full((used, len(y)), np.nan)
        for r, train, test in outer:
            window_p['z'][r, test] = C.fit_predict(Xc, y, fit_groups, ['z'], train, test)[0]
            prior[r, test] = y[train].mean()
            for k, base in bases.items():
                window_p[k][r, test] = C.fit_predict(Xc, y, fit_groups, base, train, test)[0]
                window_p[f'{k}+z'][r, test], w = stacked(Xc, y, fit_groups, base, train, test)
                weights[k].append(w)
                fold = dict(Xc, resid=residual_z(Xc, base, train))
                window_p[f'{k}|z'][r, test] = C.fit_predict(fold, y, fit_groups, ['resid'], train, test)[0]
            print(cohort, label, 'outer fold done', r, flush=True)
        unit_p = {}
        for n in rows:
            keys, unit_label, unit_p[n] = C.aggregate(cohort, y, groups, animals, window_p[n], used)
        _, _, null = C.aggregate(cohort, y, groups, animals, prior, used)
        entry = dict(windows=int(len(ix)), unit_keys=keys.tolist(), unit_labels=unit_label.tolist(),
                     scores={n: dict(C.summary(unit_label, unit_p[n]), log_loss_bits=log_loss_bits(unit_label, unit_p[n]))
                             for n in rows},
                     null_log_loss_bits=log_loss_bits(unit_label, null), bases={}, units={n: unit_p[n].tolist() for n in rows})
        for k in bases:
            d = {n: entry['scores'][n]['log_loss_bits'] for n in (k, 'z', f'{k}+z')}
            u_z, u_x = d[k] - d[f'{k}+z'], d['z'] - d[f'{k}+z']
            entry['bases'][k] = dict(
                blocks=bases[k],
                fused_minus_base=dict(auroc_difference=entry['scores'][f'{k}+z']['auroc'] - entry['scores'][k]['auroc'],
                                      interval=C.paired_bootstrap(unit_label, unit_p[f'{k}+z'], unit_p[k], keys)),
                conditional_auroc_interval=bootstrap_auroc(unit_label, unit_p[f'{k}|z'], keys),
                meta_weights_mean=np.mean(weights[k], axis=0).tolist(),
                commonality_bits=dict(unique_z=u_z, unique_base=u_x,
                                      shared=entry['null_log_loss_bits'] - d[f'{k}+z'] - u_z - u_x))
            print(cohort, label, k, entry['bases'][k]['commonality_bits'], flush=True)
        report['contrasts'][label] = entry
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=sorted(C.CONTRASTS))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    a = parser.parse_args()
    main(a.cohort, a.root, a.output, a.repeats)
