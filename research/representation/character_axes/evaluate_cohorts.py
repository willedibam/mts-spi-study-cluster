"""Cohort evaluation: grouped cross-validated prediction and surrogate-decomposed axis effects.

Prediction: participant-grouped, class-stratified five-fold CV repeated five
times; blocks are fitted inside training folds only; C is chosen by inner
grouped CV. Units are participants (or participant x condition when every
participant contributes both classes); unit probability = mean window
probability, averaged over repeats. Effects use unit-level axis means for real
windows, their IAAFT surrogates, and real - surrogate.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from research.representation.character_axes.evaluate_banks import PREDICTIVE, Blocks, readout
from research.representation.dataset_screen.common import folds

CONTRASTS = dict(ad_ftd=[('AD_vs_control', 1, 0), ('FTD_vs_control', 2, 0), ('AD_vs_FTD', 1, 2)],
                 lemon=[('old_vs_young', 1, 0)],
                 propofol=[('moderate_vs_baseline', 2, 0), ('mild_vs_baseline', 1, 0), ('moderate_vs_recovery', 2, 3)],
                 neurotycho=[('anaesthesia_vs_awake', 1, 0)])
SETS = dict(power=['power'], spectral=['spectral'], pearson=['pearson'], z=['z'], axes=['axes'], axes_levels=['axes', 'levels'],
            pearson_axes=['pearson', 'axes'], pearson_axes_levels=['pearson', 'axes', 'levels'], power_axes=['power', 'axes'],
            power_pearson=['power', 'pearson'], power_pearson_axes=['power', 'pearson', 'axes', 'levels'])
PRIMARY = [('axes_levels', 'pearson'), ('pearson_axes_levels', 'pearson'), ('power_axes', 'power'), ('power_pearson_axes', 'power_pearson')]
FRONTAL, PARIETAL = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'], ['P7', 'P3', 'Pz', 'P4', 'P8']


def features(root):
    bank = np.load(root / 'bank.npz')
    r = readout(bank)
    X = dict(spectral=bank['spectral'], pearson=bank['pearson'], z=bank['z'], axes=r['axes'], levels=r['levels'])
    power_path = root / 'absolute-power.npz'
    if power_path.exists():
        p = np.load(power_path)
        if list(p['row_id'].astype(str)) != list(bank['row_id'].astype(str)):
            raise ValueError('power rows misaligned')
        X['power'] = p['power']
    return bank, r, X


def fit_predict(X, y, groups, names, train, test):
    sub = {k: v[train] for k, v in X.items()}
    best = None
    for C in (.01, .1, 1, 10):
        p = np.zeros(len(train))
        for a, b in folds(y[train], groups[train]):
            inner = {k: v[a] for k, v in sub.items()}
            blocks = Blocks(names).fit(inner)
            model = LogisticRegression(C=C, max_iter=5000).fit(blocks.transform(inner), y[train][a])
            p[b] = model.predict_proba(blocks.transform({k: v[b] for k, v in sub.items()}))[:, 1]
        loss = np.mean((p - y[train]) ** 2)
        if best is None or loss < best[0]:
            best = (loss, C)
    blocks = Blocks(names).fit(sub)
    model = LogisticRegression(C=best[1], max_iter=5000).fit(blocks.transform(sub), y[train])
    return model.predict_proba(blocks.transform({k: v[test] for k, v in X.items()}))[:, 1]


def unit_labels(y, groups):
    within = all(len(np.unique(y[groups == g])) > 1 for g in np.unique(groups))
    return np.array([f'{g}|{c}' for g, c in zip(groups, y)]) if within else groups


def cross_validate(X, y, groups, names, repeats=5):
    units = unit_labels(y, groups)
    keys = np.unique(units)
    probs = np.zeros((repeats, len(keys)))
    for r in range(repeats):
        p = np.zeros(len(y))
        for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=261102 + r).split(y, y, groups):
            p[test] = fit_predict(X, y, groups, names, train, test)
        probs[r] = [p[units == k].mean() for k in keys]
    label = np.array([y[units == k][0] for k in keys])
    return keys, label, probs.mean(0)


def summary(label, p):
    return dict(auroc=float(roc_auc_score(label, p)), brier=float(np.mean((p - label) ** 2)),
                balanced_accuracy=float(np.mean([np.mean((p[label == c] > .5) == c) for c in (0, 1)])))


def paired_bootstrap(label, a, b, keys, n=2000, seed=261102):
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


def unit_means(values, y, groups):
    units = unit_labels(y, groups)
    keys = np.unique(units)
    return keys, np.array([y[units == k][0] for k in keys]), np.array([np.nanmean(values[units == k], 0) for k in keys])


def effect(values, y, groups):
    keys, label, means = unit_means(values, y, groups)
    within = '|' in keys[0]
    out = []
    for k in range(means.shape[1]):
        if within:
            people = np.array([u.split('|')[0] for u in keys])
            common = sorted(set(people[label == 1]) & set(people[label == 0]))
            d = np.array([means[(people == p) & (label == 1), k][0] - means[(people == p) & (label == 0), k][0] for p in common])
            d = d[np.isfinite(d)]
            test = stats.wilcoxon(d) if len(d) > 5 and np.any(d) else None
            out.append(dict(difference=float(np.mean(d)), standardized=float(np.mean(d) / np.std(d, ddof=1)) if len(d) > 2 and np.std(d) > 0 else None,
                            positive=int(np.sum(d > 0)), units=len(d), p=float(test.pvalue) if test else None))
        else:
            a, c = means[label == 1, k], means[label == 0, k]
            a, c = a[np.isfinite(a)], c[np.isfinite(c)]
            pooled = np.sqrt((np.var(a, ddof=1) + np.var(c, ddof=1)) / 2)
            out.append(dict(difference=float(a.mean() - c.mean()), standardized=float((a.mean() - c.mean()) / pooled) if pooled > 0 else None,
                            units=[len(a), len(c)], p=float(stats.mannwhitneyu(a, c).pvalue)))
    return out


def directionality(aligned, M_bank, spi_order, y, groups):
    """Frontal->parietal asymmetry of PSI and TE per window, at edge level (channel identity kept)."""
    spi = [str(s) for s in spi_order]
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    M = len(channels)
    if int(M_bank) != M:
        return {}
    off = [(i, j) for i in range(M) for j in range(M) if i != j]
    position = {e: k for k, e in enumerate(off)}
    A = aligned.reshape(len(y), len(spi), len(off))
    fronts, backs = [channels.index(c) for c in FRONTAL], [channels.index(c) for c in PARIETAL]
    result = {}
    for name in ['psi_multitaper_mean_fs-1_fmin-0_fmax-0-25', 'psi_multitaper_mean_fs-1_fmin-0_fmax-0-5',
                 'te_kraskov_NN-4_k-1_kt-1_l-1_lt-1', 'gc_gaussian_k-1_kt-1_l-1_lt-1']:
        if name not in spi:
            continue
        k = spi.index(name)
        forward = np.nanmean(A[:, k, [position[(i, j)] for i in fronts for j in backs]], 1)
        backward = np.nanmean(A[:, k, [position[(j, i)] for i in fronts for j in backs]], 1)
        index = forward - backward
        result[name] = dict(asymmetry_by_class={int(c): float(np.nanmean(index[y == c])) for c in np.unique(y)},
                            effect=effect(index[:, None], y, groups)[0])
    return result


def main(cohort, root, surrogate_root, output):
    bank, r, X = features(root)
    y_all, groups_all = bank['y'], bank['group'].astype(str)
    names = PREDICTIVE + [f'level_{n}' for n in r['level_names']]
    real = np.hstack([r['axes'], r['levels']])
    decomposition = None
    if surrogate_root is not None and (surrogate_root / 'bank.npz').exists():
        sb = np.load(surrogate_root / 'bank.npz')
        rs = readout(sb)
        source = [s.removeprefix('s-') for s in sb['row_id'].astype(str)]
        order = {n: k for k, n in enumerate(bank['row_id'].astype(str))}
        if set(source) != set(order):
            raise ValueError('surrogate rows do not match the real bank')
        surrogate = np.full_like(real, np.nan)
        surrogate[[order[s] for s in source]] = np.hstack([rs['axes'], rs['levels']])
        decomposition = surrogate
    report = dict(cohort=cohort, windows=int(len(y_all)), participants=int(len(np.unique(groups_all))),
                  identity_floor=float(np.nanmin(r['controls'][:, 0])), contrasts={})
    if cohort == 'neurotycho':
        # Original design: ketamine-medetomidine sources, propofol targets, target animal never in its fit.
        roles = bank['role'].astype(str)
        groups_all = np.array([f'{g}-{r}' for g, r in zip(bank['group'].astype(str), roles)])
        animals = bank['group'].astype(str)
    for label, positive, negative in CONTRASTS[cohort]:
        ix = np.isin(y_all, [positive, negative])
        y, groups = (y_all[ix] == positive).astype(int), groups_all[ix]
        Xc = {k: v[ix] for k, v in X.items()}
        entry = dict(windows=int(ix.sum()), scores={}, primary={}, effects={})
        predictions = {}
        for name, blocks in SETS.items():
            if not all(b in Xc for b in blocks):
                continue
            if cohort == 'neurotycho':
                p = np.full(len(y), np.nan)
                for target in np.unique(animals[roles == 'target']):
                    train = np.flatnonzero((roles == 'source') & (animals != target))
                    test = np.flatnonzero((roles == 'target') & (animals == target))
                    p[test] = fit_predict(Xc, y, animals, blocks, train, test)
                keep = np.isfinite(p)
                keys, unit_label, p = np.array([f'{a}|{k}' for k, a in enumerate(animals[keep])]), y[keep], p[keep]
            else:
                keys, unit_label, p = cross_validate(Xc, y, groups, blocks)
            predictions[name] = p
            entry['scores'][name] = summary(unit_label, p)
            print(cohort, label, name, {k: round(v, 3) for k, v in entry['scores'][name].items()}, flush=True)
        for a, b in PRIMARY:
            if a in predictions and b in predictions:
                entry['primary'][f'{a} - {b}'] = dict(auroc_difference=entry['scores'][a]['auroc'] - entry['scores'][b]['auroc'],
                                                      interval=paired_bootstrap(unit_label, predictions[a], predictions[b], keys))
        parts = dict(real=real[ix])
        if decomposition is not None:
            parts.update(surrogate=decomposition[ix], real_minus_surrogate=real[ix] - decomposition[ix])
        for part, values in parts.items():
            entry['effects'][part] = dict(zip(names, effect(values, y, groups)))
        if cohort == 'propofol':
            entry['directionality'] = directionality(bank['aligned'][ix], bank['M'][0], bank['spi_order'], y, groups)
        report['contrasts'][label] = entry
    output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=sorted(CONTRASTS))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--surrogate-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.cohort, args.root, args.surrogate_root, args.output)
