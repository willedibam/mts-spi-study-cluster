"""Comparator ladder on the character-axes cohort banks, under their frozen protocol.

Protocol (unchanged from character_axes.evaluate_cohorts): participant-grouped, class-stratified five-fold CV
repeated five times, L2 logistic with C chosen by inner grouped CV, each block imputed, scaled and PCA-capped
inside the training fold, unit probability = mean window probability. NeuroTycho keeps its transfer design
(ketamine-medetomidine sources, propofol targets, target animal never in its fit, windows pooled).

Added here: r + z, spectra + z, mean-SPI, the 23-descriptor SPI marginals m, full aligned MPIs, the best single
SPI chosen inside each training fold, tangent-space covariance, per-channel catch22, and z read through its
first PC or at full rank. The tangent reference is fitted on the outer training rows only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research.representation.comparator_fill import blocks as B
from research.representation.dataset_screen.common import folds

CONTRASTS = dict(ad_ftd=[('AD_vs_control', 1, 0), ('FTD_vs_control', 2, 0), ('AD_vs_FTD', 1, 2)],
                 lemon=[('old_vs_young', 1, 0)],
                 propofol=[('moderate_vs_baseline', 2, 0), ('mild_vs_baseline', 1, 0), ('moderate_vs_recovery', 2, 3)],
                 neurotycho=[('anaesthesia_vs_awake', 1, 0)])
SETS = dict(pearson=['pearson'], spectral=['spectral'], power=['power'], z=['z'], z_pc1=['z_pc1'], z_full=['z_full'],
            pearson_z=['pearson', 'z'], spectral_z=['spectral', 'z'], power_z=['power', 'z'],
            pearson_spectral=['pearson', 'spectral'], pearson_spectral_z=['pearson', 'spectral', 'z'],
            mean_spi=['mean_spi'], mean_spi_z=['mean_spi', 'z'], m=['m'], m_z=['m', 'z'], aligned=['aligned'],
            tangent=['tangent'], catch22=['catch22'], single_spi=['single_spi'])
PAIRS = [('pearson_z', 'pearson'), ('spectral_z', 'spectral'), ('power_z', 'power'),
         ('pearson_spectral_z', 'pearson_spectral'), ('mean_spi_z', 'mean_spi'), ('m_z', 'm'),
         ('z', 'pearson'), ('z', 'mean_spi'), ('z', 'm'), ('z', 'aligned'), ('z', 'single_spi'), ('z', 'tangent'),
         ('z', 'catch22'), ('z', 'spectral'), ('z_full', 'z'), ('z_pc1', 'z')]
CAPS = dict(z_pc1=1, z_full=None)


def steps(name, width):
    """evaluate_banks.block, except that z_pc1 keeps one component and z_full keeps every component."""
    base = [SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()]
    cap = CAPS.get(name, 32)
    if cap is None:
        return base + [PCA(random_state=261031)]
    return base + ([PCA(cap, random_state=261031)] if width > cap else [])


class Blocks:
    def __init__(self, names): self.names = names
    def fit(self, X):
        self.pipes = [make_pipeline(*steps(n, X[n].shape[1])).fit(X[n]) for n in self.names]
        return self
    def transform(self, X):
        return np.hstack([p.transform(X[n]) for p, n in zip(self.pipes, self.names)])


def fit_predict(X, y, groups, names, train, test):
    """evaluate_cohorts.fit_predict (same folds, grid and loss), also returning the inner loss and chosen C."""
    sub = {n: X[n][train] for n in names}
    inner = []
    for a, b in folds(y[train], groups[train]):
        blocks = Blocks(names).fit({n: v[a] for n, v in sub.items()})
        inner.append((a, b, blocks.transform({n: v[a] for n, v in sub.items()}), blocks.transform({n: v[b] for n, v in sub.items()})))
    best = None
    for C in (.01, .1, 1, 10):
        p = np.zeros(len(train))
        for a, b, fa, fb in inner:
            p[b] = LogisticRegression(C=C, max_iter=5000).fit(fa, y[train][a]).predict_proba(fb)[:, 1]
        loss = np.mean((p - y[train]) ** 2)
        if best is None or loss < best[0]:
            best = (loss, C)
    if test is None:
        return None, best
    blocks = Blocks(names).fit(sub)
    model = LogisticRegression(C=best[1], max_iter=5000).fit(blocks.transform(sub), y[train])
    return model.predict_proba(blocks.transform({n: X[n][test] for n in names}))[:, 1], best


def single_spi(profiles, y, groups, train, test, jobs):
    """Best single SPI by inner loss on the training rows, then that SPI's fit scored on the test rows."""
    losses = Parallel(n_jobs=jobs)(delayed(fit_predict)(dict(u=profiles[train, k]), y[train], groups[train], ['u'],
                                                        np.arange(len(train)), None) for k in range(profiles.shape[1]))
    k = int(np.argmin([loss for _, (loss, _) in losses]))
    return fit_predict(dict(u=profiles[:, k]), y, groups, ['u'], train, test)[0], k


def unit_labels(y, groups):
    within = all(len(np.unique(y[groups == g])) > 1 for g in np.unique(groups))
    return np.array([f'{g}|{c}' for g, c in zip(groups, y)]) if within else groups


def summary(label, p):
    from sklearn.metrics import roc_auc_score
    return dict(auroc=float(roc_auc_score(label, p)), brier=float(np.mean((p - label) ** 2)),
                balanced_accuracy=float(np.mean([np.mean((p[label == c] > .5) == c) for c in (0, 1)])))


def paired_bootstrap(label, a, b, keys, n=2000, seed=261102):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    people = np.array([k.split('|')[0] for k in keys])
    unique = np.unique(people)
    diffs = []
    for _ in range(n):
        pick = np.concatenate([np.flatnonzero(people == u) for u in rng.choice(unique, len(unique))])
        if len(np.unique(label[pick])) < 2:
            continue
        diffs.append(roc_auc_score(label[pick], a[pick]) - roc_auc_score(label[pick], b[pick]))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def splits(cohort, y, groups, roles, animals, repeats):
    """Outer (repeat, train, test) index triples: the cohort CV, or NeuroTycho's leave-target-animal-out transfer."""
    if cohort == 'neurotycho':
        return [(0, np.flatnonzero((roles == 'source') & (animals != t)), np.flatnonzero((roles == 'target') & (animals == t)))
                for t in np.unique(animals[roles == 'target'])]
    return [(r, train, test) for r in range(repeats)
            for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=261102 + r).split(y, y, groups)]


def aggregate(cohort, y, groups, animals, window_p, repeats):
    """Window probabilities per repeat -> unit keys, labels and probabilities, as the original scripts."""
    if cohort == 'neurotycho':
        keep = np.isfinite(window_p[0])
        return np.array([f'{a}|{k}' for k, a in enumerate(animals[keep])]), y[keep], window_p[0][keep]
    units = unit_labels(y, groups)
    keys = np.unique(units)
    probs = np.array([[window_p[r][units == k].mean() for k in keys] for r in range(repeats)])
    return keys, np.array([y[units == k][0] for k in keys]), probs.mean(0)


def windows(root, bank):
    """Raw windows (n, T, M) in bank row order."""
    if 'x' in bank.files:
        return np.asarray(bank['x'], dtype=np.float64)
    rows = json.loads((root / 'manifest.json').read_text())['rows']
    if [r['row_id'] for r in rows] != list(bank['row_id'].astype(str)):
        raise ValueError('raw windows misaligned')
    with np.load(root / 'views.npz', allow_pickle=False) as views:
        return np.stack([views[r['row_id']] for r in rows]).astype(np.float64)


def load(root):
    bank = np.load(root / 'bank.npz')
    X = {k: np.asarray(bank[k]) for k in ('spectral', 'pearson', 'z', 'm', 'aligned') if k in bank.files}
    power = root / 'absolute-power.npz'
    if power.exists():
        p = np.load(power)
        if list(p['row_id'].astype(str)) != list(bank['row_id'].astype(str)):
            raise ValueError('power rows misaligned')
        X['power'] = p['power']
    x = windows(root, bank)
    spi = len(bank['spi_order'])
    X.update(z_pc1=X['z'], z_full=X['z'], mean_spi=B.mean_spi(X['aligned'], spi), catch22=B.catch22(x))
    return bank, X, x, X['aligned'].reshape(len(x), spi, -1)


def main(cohort, root, output, repeats, jobs, only):
    bank, X, x, profiles = load(root)
    y_all, groups_all = bank['y'], bank['group'].astype(str)
    roles_all = bank['role'].astype(str) if 'role' in bank.files else np.full(len(y_all), '')
    animals_all = groups_all
    if cohort == 'neurotycho':
        groups_all = np.array([f'{g}-{r}' for g, r in zip(groups_all, roles_all)])
    names = [n for n in (only or SETS) if all(b in X or b in ('tangent', 'single_spi') for b in SETS[n])]
    report = dict(cohort=cohort, windows=int(len(y_all)), spi=[str(s) for s in bank['spi_order']], contrasts={})
    for label, positive, negative in CONTRASTS[cohort]:
        ix = np.flatnonzero(np.isin(y_all, [positive, negative]))
        y, groups, roles, animals = (y_all[ix] == positive).astype(int), groups_all[ix], roles_all[ix], animals_all[ix]
        Xc = {k: v[ix] for k, v in X.items()}
        fit_groups = animals if cohort == 'neurotycho' else groups
        outer = splits(cohort, y, groups, roles, animals, repeats)
        used = max(r for r, _, _ in outer) + 1
        window_p = {n: np.full((used, len(y)), np.nan) for n in names}
        chosen = []
        for r, train, test in outer:
            fold = dict(Xc)
            if 'tangent' in names:
                fold['tangent'] = B.tangent(x[ix], train)
            for n in names:
                if n == 'single_spi':
                    window_p[n][r, test], k = single_spi(profiles[ix], y, fit_groups, train, test, jobs)
                    chosen.append(report['spi'][k])
                else:
                    window_p[n][r, test] = fit_predict(fold, y, fit_groups, SETS[n], train, test)[0]
            print(cohort, label, 'outer fold done', r, len(test), flush=True)
        entry = dict(windows=int(len(ix)), scores={}, contrasts={}, single_spi_selected=chosen, units={})
        unit_p = {}
        for n in names:
            keys, unit_label, unit_p[n] = aggregate(cohort, y, groups, animals, window_p[n], used)
            entry['scores'][n] = summary(unit_label, unit_p[n])
            entry['units'][n] = unit_p[n].tolist()
            print(cohort, label, n, {k: round(v, 3) for k, v in entry['scores'][n].items()}, flush=True)
        entry.update(unit_keys=keys.tolist(), unit_labels=unit_label.tolist())
        for a, b in PAIRS:
            if a in unit_p and b in unit_p:
                entry['contrasts'][f'{a} - {b}'] = dict(
                    auroc_difference=entry['scores'][a]['auroc'] - entry['scores'][b]['auroc'],
                    interval=paired_bootstrap(unit_label, unit_p[a], unit_p[b], keys))
        report['contrasts'][label] = entry
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=sorted(CONTRASTS))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--only', nargs='*')
    a = parser.parse_args()
    main(a.cohort, a.root, a.output, a.repeats, a.jobs, a.only)
