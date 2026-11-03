"""Mechanism test: does a widespread common component decouple pairwise from direct dependence?

Under anaesthesia the edge pattern of direct dependence (partial r^2) stops tracking that of pairwise dependence
(r^2, rank and kernel dependence; direct_marginal.py). If a common drive adds pairwise dependence to every edge
irrespective of direct coupling, then removing the common component (global mean signal, or the first principal
component) from each window should restore alignment. Per window (channels standardized): alignment = Pearson over
edges of partial r^2 with r^2 of the raw, global-signal-regressed and PC1-removed data; plus common-mode share
(top eigenvalue / M), mean r^2 and mean partial r^2. Effects for real data, both IAAFT variants and real - surrogate.
Run on Gadi (jobs/common_drive.pbs).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from research.representation.character_axes.direct_marginal import COHORTS, units
from research.representation.character_axes.evaluate_cohorts import CONTRASTS, effect

NAMES = ['align_raw', 'align_gsr', 'align_pc1', 'restored_gsr', 'restored_pc1', 'common_share', 'mean_r2', 'mean_r2_gsr', 'mean_partial_r2']


def window(x):
    x = (x - x.mean(0)) / x.std(0)
    M = x.shape[1]
    off = ~np.eye(M, dtype=bool)
    c = np.corrcoef(x, rowvar=False)
    p = np.linalg.pinv(c)
    d = np.sqrt(np.abs(np.diag(p)))
    pr2 = (p / np.outer(d, d))[off] ** 2
    g = x.mean(1, keepdims=True)
    xg = x - g @ (g.T @ x) / (g.T @ g)
    w, v = np.linalg.eigh(c)
    xp = x - (x @ v[:, -1:]) @ v[:, -1:].T
    r2 = c[off] ** 2
    r2g = np.corrcoef(xg, rowvar=False)[off] ** 2
    r2p = np.corrcoef(xp, rowvar=False)[off] ** 2
    a = [np.corrcoef(pr2, r)[0, 1] for r in (r2, r2g, r2p)]
    return [a[0], a[1], a[2], a[1] - a[0], a[2] - a[0], w[-1] / M, r2.mean(), r2g.mean(), pr2.mean()]


def readouts(bank):
    return np.array([window(np.asarray(x, np.float64)) for x in bank['x']])


def main(root, output):
    report = dict(names=NAMES, cohorts={})
    f = lambda e: f"{e['standardized']:+6.2f} {e.get('positive', '')}/{e['units']} p{e['p']:.3f}" if e['standardized'] is not None else 'n/a'
    for cohort, (rel, srel, variants) in COHORTS.items():
        bank = np.load(root / rel / 'bank.npz')
        X, y, g = readouts(bank), bank['y'], units(bank, cohort)
        sur = {v: readouts(np.load(root / srel.format(v) / 'bank.npz')) for v in variants}
        rep = dict(contrasts={}, class_means={int(c): np.nanmean(X[y == c], 0).tolist() for c in np.unique(y)},
                   surrogate_class_means={v: {int(c): np.nanmean(S[y == c], 0).tolist() for c in np.unique(y)} for v, S in sur.items()})
        for label, pos, neg in CONTRASTS[cohort]:
            ix = np.isin(y, [pos, neg])
            yy = (y[ix] == pos).astype(int)
            rep['contrasts'][label] = dict(real=effect(X[ix], yy, g[ix]), **{f'surr{v}': effect(S[ix], yy, g[ix]) for v, S in sur.items()},
                                           **{f'resid{v}': effect(X[ix] - S[ix], yy, g[ix]) for v, S in sur.items()})
            print(cohort, label, flush=True)
            for k, n in enumerate(NAMES):
                r = rep['contrasts'][label]
                print(f'  {n:16s} real {f(r["real"][k])}', *(f'| surr{v} {f(r[f"surr{v}"][k])} | res {f(r[f"resid{v}"][k])}' for v in sur), flush=True)
        print(cohort, 'class means', {c: [round(v, 3) for v in m] for c, m in rep['class_means'].items()}, flush=True)
        print(cohort, 'surrogate class means', {v: {c: [round(u, 3) for u in m] for c, m in s.items()} for v, s in rep['surrogate_class_means'].items()}, flush=True)
        report['cohorts'][cohort] = rep
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, default=float) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.output)
