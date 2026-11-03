"""Mechanism test: is coupling organised differently during large joint excursions than in the bulk of samples?

Robust (MinCovDet) and classical covariance estimates diverge in their edge pattern under anaesthesia beyond what
spectra-preserving surrogates reproduce (direct_marginal.py). MinCovDet down-weights samples with large Mahalanobis
distance, so the hypothesis is state-dependent coupling: the network couples one way during large joint excursions
(e.g. slow-wave up/down transitions) and another way in between. Per window (channels standardized): samples are split
into events (top quarter of Mahalanobis distance) and bulk; per subset, correlation and partial-correlation matrices.
Readouts: similarity over edges of event and bulk patterns (partial r^2, r^2), the same for a random quarter split
(finite-sample reference), their difference, and the common-mode share (top eigenvalue / M) in events and bulk.
Effects for real data, both IAAFT variants and real - surrogate. Run on Gadi (jobs/event_coupling.pbs).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from research.representation.character_axes.direct_marginal import COHORTS, units
from research.representation.character_axes.evaluate_cohorts import CONTRASTS, effect

Q = 0.25
NAMES = ['sim_partial_event', 'sim_partial_random', 'sim_partial_gap', 'sim_marginal_event', 'sim_marginal_random',
         'sim_marginal_gap', 'common_event', 'common_bulk', 'common_gap', 'strength_event', 'strength_bulk', 'strength_gap']


def mats(x):
    c = np.corrcoef(x, rowvar=False)
    p = np.linalg.pinv(c)
    d = np.sqrt(np.abs(np.diag(p)))
    return c, -p / np.outer(d, d)


def window(x, rng):
    x = (x - x.mean(0)) / x.std(0)
    M = x.shape[1]
    off = ~np.eye(M, dtype=bool)
    m2 = np.einsum('ti,ij,tj->t', x, np.linalg.pinv(np.cov(x, rowvar=False)), x)
    ev = m2 >= np.quantile(m2, 1 - Q)
    rd = np.zeros(len(x), bool)
    rd[rng.choice(len(x), ev.sum(), replace=False)] = True
    out = {}
    for tag, mask in (('event', ev), ('random', rd)):
        (ca, pa), (cb, pb) = mats(x[mask]), mats(x[~mask])
        out[f'sim_partial_{tag}'] = np.corrcoef(pa[off] ** 2, pb[off] ** 2)[0, 1]
        out[f'sim_marginal_{tag}'] = np.corrcoef(ca[off] ** 2, cb[off] ** 2)[0, 1]
        if tag == 'event':
            out['common_event'], out['common_bulk'] = (np.linalg.eigvalsh(ca)[-1] / M, np.linalg.eigvalsh(cb)[-1] / M)
            out['strength_event'], out['strength_bulk'] = np.mean(ca[off] ** 2), np.mean(cb[off] ** 2)
    out['sim_partial_gap'] = out['sim_partial_event'] - out['sim_partial_random']
    out['sim_marginal_gap'] = out['sim_marginal_event'] - out['sim_marginal_random']
    out['common_gap'] = out['common_event'] - out['common_bulk']
    out['strength_gap'] = out['strength_event'] - out['strength_bulk']
    return [out[n] for n in NAMES]


def readouts(bank):
    rng = np.random.default_rng(261103)
    return np.array([window(np.asarray(x, np.float64), rng) for x in bank['x']])


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
                print(f'  {n:20s} real {f(r["real"][k])}', *(f'| res{v} {f(r[f"resid{v}"][k])}' for v in sur), flush=True)
        print(cohort, 'class means', {c: [round(v, 3) for v in m] for c, m in rep['class_means'].items()}, flush=True)
        report['cohorts'][cohort] = rep
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, default=float) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.output)
