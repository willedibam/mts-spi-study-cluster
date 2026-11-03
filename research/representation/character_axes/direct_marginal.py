"""Exploration: does the alignment between direct (partial) and pairwise dependence change across states and cohorts?

Per window, z(a|b) = Pearson correlation over edges of SPI a's and SPI b's edge profiles (as in the bank's z), computed
here for chosen pairs, plus each SPI's level (median over edges). Effects per cohort contrast for real data and both
IAAFT surrogate variants (residual = real - surrogate). Propofol also gets unit means at all four levels for a
dose-response read. Run on Gadi (jobs/direct_marginal.pbs).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mtsspi.character_axes import edge_profiles, profile_correlation
from research.representation.character_axes.evaluate_cohorts import CONTRASTS, effect

EST = ('MinCovDet', 'EllipticEnvelope', 'EmpiricalCovariance', 'LedoitWolf', 'OAS', 'ShrunkCovariance', 'GraphicalLasso')
PAIRS = [(f'prec-sq_{e}', m) for e in EST for m in ('spearmanr-sq', 'ids', f'cov-sq_{e}', 'mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO')]
PAIRS += [('prec-sq_MinCovDet', 'prec-sq_EmpiricalCovariance'), ('cov-sq_MinCovDet', 'cov-sq_EmpiricalCovariance'),
          ('cov-sq_EmpiricalCovariance', 'spearmanr-sq'), ('cov-sq_EmpiricalCovariance', 'ids'), ('spearmanr-sq', 'ids')]
LEVELS = sorted({s for p in PAIRS for s in p})
COHORTS = dict(neurotycho=('character-transfer-261031/neurotycho-primary', 'character-axes-261102/surr-neurotycho{}', ('-marginal1000', '-spectrum')),
               propofol=('character-axes-261102/propofol', 'character-axes-261102/surr-propofol{}', ('-marginal', '-spectrum')),
               lemon=('character-axes-261102/lemon', 'character-axes-261102/surr-lemon{}', ('-marginal', '-spectrum')),
               ad_ftd=('character-axes-261102/ad-ftd', 'character-axes-261102/surr-ad-ftd{}', ('-marginal', '-spectrum')))


def readouts(bank):
    spi = [str(s) for s in bank['spi_order']]
    P = edge_profiles(bank['aligned'], int(bank['M'][0]), len(spi))
    z = np.column_stack([profile_correlation(P[:, spi.index(a)], P[:, spi.index(b)]) for a, b in PAIRS])
    lv = np.column_stack([np.nanmedian(P[:, spi.index(s)], 1) for s in LEVELS])
    return np.column_stack([z, lv])


def units(bank, cohort):
    g = bank['group'].astype(str)
    return np.array([f'{a}-{r}' for a, r in zip(g, bank['role'].astype(str))]) if cohort == 'neurotycho' else g


def main(root, output):
    names = [f'z {a}|{b}' for a, b in PAIRS] + [f'level {s}' for s in LEVELS]
    report = dict(names=names, cohorts={})
    for cohort, (rel, srel, variants) in COHORTS.items():
        bank = np.load(root / rel / 'bank.npz')
        X, y, g = readouts(bank), bank['y'], units(bank, cohort)
        sur = {}
        for v in variants:
            p = root / srel.format(v) / 'bank.npz'
            if p.exists():
                sb = np.load(p)
                assert list(sb['row_id'].astype(str)) == list(bank['row_id'].astype(str)), p
                sur[v] = readouts(sb)
        rep = dict(contrasts={}, surrogates=list(sur))
        for label, pos, neg in CONTRASTS[cohort]:
            ix = np.isin(y, [pos, neg])
            yy = (y[ix] == pos).astype(int)
            rep['contrasts'][label] = dict(real=effect(X[ix], yy, g[ix]),
                                           **{f'resid{v}': effect(X[ix] - S[ix], yy, g[ix]) for v, S in sur.items()},
                                           **{f'surr{v}': effect(S[ix], yy, g[ix]) for v, S in sur.items()})
        if cohort == 'propofol':
            rep['dose'] = {int(c): np.nanmean(np.stack([np.nanmean(X[(g == u) & (y == c)], 0) for u in np.unique(g)]), 0).tolist() for c in np.unique(y)}
        report['cohorts'][cohort] = rep
        f = lambda e: f"{e['standardized']:+6.2f}" if e['standardized'] is not None else '   n/a'
        for label, r in rep['contrasts'].items():
            print(cohort, label, flush=True)
            for k, n in enumerate(names):
                print(f'  {n[:62]:62s} real', f(r['real'][k]), *(f'res{v}' + f(r[f"resid{v}"][k]) for v in sur), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, default=float) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.root, a.output)
