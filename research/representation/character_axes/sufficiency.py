"""Is the construction-validated core a sufficient readout? The two checks fixed in protocol.md (3 November).

1. Extended readouts: state effects of the 13 axes and their levels after regressing them on core
   readouts plus strength covariates (slopes pooled within class, window level).
2. z's increment: cross-validated AUROC of Pearson+strength (P), P+z, P+core and P+core+z.

Run on Gadi (jobs/sufficiency.pbs), one cohort per job.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mtsspi.character_axes import CORE, core_scores, edge_profiles, masked_correlation
from research.representation.character_axes.evaluate_banks import PREDICTIVE, readout
from research.representation.character_axes.evaluate_cohorts import (CONTRASTS, cross_validate, effect, fit_predict,
                                                                       paired_bootstrap, summary)

SETS = dict(P=['pearson', 'strength'], P_z=['pearson', 'strength', 'z'], P_core=['pearson', 'strength', 'core'],
            P_core_z=['pearson', 'strength', 'core', 'z'])


def blocks(bank):
    spi = [str(s) for s in bank['spi_order']]
    M = int(bank['M'][0])
    profiles = edge_profiles(bank['aligned'], M, len(spi))
    core, core_names = core_scores(profiles, spi, M)
    r = readout(bank)
    extended = np.hstack([r['axes'], r['levels']])
    extended_names = PREDICTIVE + [f'level_{n}' for n in r['level_names']]
    pearson = np.abs(np.asarray(bank['pearson'], dtype=np.float64))
    strength_range = masked_correlation(profiles[:, spi.index('cov-sq_EmpiricalCovariance')].astype(np.float64),
                                        profiles[:, spi.index('mi_gaussian')].astype(np.float64))
    strength = np.column_stack([np.nanmean(pearson, 1), np.nanstd(pearson, 1), strength_range])
    X = dict(pearson=np.asarray(bank['pearson']), strength=strength, core=core, z=np.asarray(bank['z']))
    return X, extended, extended_names, core_names


def adjust(values, covariates, y):
    """Residualise each column on covariates with slopes pooled within class (ANCOVA-style), keeping class means."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    ok_z = np.all(np.isfinite(covariates), 1)
    for k in range(values.shape[1]):
        ok = ok_z & np.isfinite(values[:, k])
        if ok.sum() < covariates.shape[1] + 5:
            continue
        Z, x, c = covariates[ok], values[ok, k], y[ok]
        Zw, xw = Z.copy(), x.copy()
        for label in np.unique(c):
            Zw[c == label] -= Z[c == label].mean(0)
            xw[c == label] -= x[c == label].mean()
        scale = Zw.std(0)
        scale[scale < 1e-12] = 1
        beta = np.linalg.lstsq(Zw / scale, xw, rcond=None)[0]
        out[ok, k] = x - ((Z - Z.mean(0)) / scale) @ beta
    return out


def verdict(raw, adjusted):
    d, a = raw['standardized'], adjusted['standardized']
    if d is None or abs(d) < .5 or raw['p'] is None or raw['p'] >= .05:
        return 'not eligible'
    if a is not None and np.sign(a) == np.sign(d) and abs(a) >= .5 and adjusted['p'] is not None and adjusted['p'] < .05:
        return 'survives'
    return 'absorbed' if a is None or abs(a) <= .5 * abs(d) else 'partial'


def predict(cohort, Xc, y, groups, names, bank, ix):
    if cohort != 'neurotycho':
        return cross_validate(Xc, y, groups, names)
    roles, animals = bank['role'].astype(str)[ix], bank['group'].astype(str)[ix]
    p = np.full(len(y), np.nan)
    for target in np.unique(animals[roles == 'target']):
        train = np.flatnonzero((roles == 'source') & (animals != target))
        test = np.flatnonzero((roles == 'target') & (animals == target))
        p[test] = fit_predict(Xc, y, animals, names, train, test)
    keep = np.isfinite(p)
    return np.array([f'{a}|{k}' for k, a in enumerate(animals[keep])]), y[keep], p[keep]


def main(cohort, root, output):
    bank = np.load(root / 'bank.npz')
    X, extended, extended_names, core_names = blocks(bank)
    y_all, groups_all = bank['y'], bank['group'].astype(str)
    if cohort == 'neurotycho':
        groups_all = np.array([f'{g}-{r}' for g, r in zip(groups_all, bank['role'].astype(str))])
    covariates = np.hstack([X['core'], X['strength']])
    report = dict(cohort=cohort, core=core_names, core_roles={n: CORE[n][0] for n in core_names}, contrasts={})
    for label, positive, negative in CONTRASTS[cohort]:
        ix = np.isin(y_all, [positive, negative])
        y, groups = (y_all[ix] == positive).astype(int), groups_all[ix]
        raw = effect(extended[ix], y, groups)
        adjusted = effect(adjust(extended[ix], covariates[ix], y), y, groups)
        readouts = {n: dict(raw=r, adjusted=a, verdict=verdict(r, a)) for n, r, a in zip(extended_names, raw, adjusted)}
        survivors = [n for n, v in readouts.items() if v['verdict'] == 'survives']
        entry = dict(windows=int(ix.sum()), core_effects=dict(zip(core_names, effect(X['core'][ix], y, groups))),
                     readouts=readouts, survivors=survivors, sufficient_for_extended=not survivors, scores={})
        Xc = {k: v[ix] for k, v in X.items()}
        predictions = {}
        for name, names in SETS.items():
            keys, unit_label, p = predict(cohort, Xc, y, groups, names, bank, ix)
            predictions[name] = p
            entry['scores'][name] = summary(unit_label, p)
            print(cohort, label, name, round(entry['scores'][name]['auroc'], 3), flush=True)
        auc = {k: v['auroc'] for k, v in entry['scores'].items()}
        dz = dict(difference=auc['P_z'] - auc['P'], interval=paired_bootstrap(unit_label, predictions['P_z'], predictions['P'], keys))
        dz_core = dict(difference=auc['P_core_z'] - auc['P_core'],
                       interval=paired_bootstrap(unit_label, predictions['P_core_z'], predictions['P_core'], keys))
        dcore = dict(difference=auc['P_core'] - auc['P'], interval=paired_bootstrap(unit_label, predictions['P_core'], predictions['P'], keys))
        exists = dz['interval'][0] > 0
        entry['z_increment'] = dict(z_over_P=dz, z_over_P_core=dz_core, core_over_P=dcore, z_adds=exists,
                                    core_accounts=(exists and dz_core['interval'][0] <= 0 <= dz_core['interval'][1]
                                                   and dz_core['difference'] <= .5 * dz['difference']) if exists else None)
        print(cohort, label, 'survivors', survivors, 'z increment', entry['z_increment'], flush=True)
        report['contrasts'][label] = entry
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, default=float) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=sorted(CONTRASTS))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    main(a.cohort, a.root, a.output)
