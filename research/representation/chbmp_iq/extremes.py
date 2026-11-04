"""Post-hoc extreme-group view of the CHBMP IQ null (not pre-registered).

Top against bottom FSIQ fractions of the E1 sample (N = 135). Each block (z, channel spectra, Pearson) is residualised on
the E1 covariates over all 135 subjects, then compared between the two extreme groups by Cohen's d per feature.
Nulls permute group labels within the extreme set (exact when there are at most 20,000 splits):
- max |d| over features (family-wise),
- RMS d over features, a whole-profile "are the groups different at all" statistic.
Extreme-group d overstates the population effect (Preacher et al. 2005, Psychol Methods 10:178), so it is read only against its
own null.
"""
import argparse
from itertools import combinations
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from mtsspi.figures import save, style
from mtsspi.spi_spi_contract import build_unified_schema
from research.representation.chbmp_iq.analysis import contrast, design, load, profiles, residual, subject_mean

FRACTIONS, PERMS = (.05, .10, .25), 5000


def splits(n, rng):
    """Boolean membership of the 'high' group over 2n subjects: every split if few, else random ones."""
    if exact(n):
        out = np.zeros((comb(2 * n, n), 2 * n), bool)
        for i, c in enumerate(combinations(range(2 * n), n)):
            out[i, list(c)] = True
        return out
    return np.stack([rng.permutation(2 * n) < n for _ in range(PERMS)])


def exact(n):
    return comb(2 * n, n) <= 20000


def cohen_d(A, H):
    """d (high minus low) for every row of H (perms x 2n membership) and column of A (2n x features)."""
    H = H.astype(float)
    n = H.sum(1, keepdims=True); m = H.shape[1] - n
    s1, q1 = H @ A, H @ A ** 2
    s0, q0 = A.sum(0) - s1, (A ** 2).sum(0) - q1
    m1, m0 = s1 / n, s0 / m
    var = (q1 - n * m1 ** 2 + q0 - m * m0 ** 2) / (n + m - 2)
    return (m1 - m0) / np.sqrt(var)


def compare(A, high, rng):
    """Observed d per feature, its max-|d| FWER, and RMS-d whole-profile p, against label splits within the extreme set."""
    d = cohen_d(A, high[None])[0]
    k = int(high.sum())
    H = splits(k, rng) if exact(k) else np.vstack([high[None], splits(k, rng)])  # the observed split counts once
    mx, rms = [], []
    for chunk in np.array_split(H, max(1, len(H) // 250)):
        D = cohen_d(A, chunk)
        mx.append(np.abs(D).max(1)); rms.append(np.sqrt((D ** 2).mean(1)))
    mx, rms = np.concatenate(mx), np.concatenate(rms)
    fwer = (np.sum(mx[None, :] >= np.abs(d)[:, None] - 1e-12, 1)) / len(mx)  # the observed split is among the splits
    obs_rms = np.sqrt((d ** 2).mean())
    return d, fwer, dict(splits=len(H), exact=exact(k),
                         max_abs_d=float(np.abs(d).max()), null_max_abs_d_95=float(np.quantile(mx, .95)),
                         fwer_significant=int((fwer < .05).sum()), rms_d=float(obs_rms),
                         null_rms_d_median=float(np.median(rms)), p_rms=float(np.mean(rms >= obs_rms - 1e-12))), mx, rms


def main(root, scope, wais, out):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(261105)
    bank, meta, power, sc = load(root, scope)
    _, order, _ = profiles(bank)
    group = bank['group']
    subs, Z = subject_mean(bank['z'].astype(float), group)
    _, S = subject_mean(power, group); _, Pe = subject_mean(bank['pearson'].astype(float), group)
    names = [(f.spi_a, f.spi_b) for f in build_unified_schema(order)]
    first = meta.groupby('group').first().loc[subs]
    w = pd.read_csv(wais, skiprows=1, encoding='utf-8-sig'); w.columns = [c.strip() for c in w.columns]
    w = w.dropna(subset=['Code'])
    fsiq = pd.to_numeric(w.assign(subject='sub-' + w.Code.astype(str).str.strip()).set_index('subject').FSIQ, errors='coerce')
    df = pd.DataFrame(dict(subject=subs, fsiq=fsiq.reindex(subs).values, age=first.age.values, sex=first.sex.values,
                           montage=first.montage.astype(str).values, education=first.education.values,
                           iaf=sc.iaf.reindex(subs).values, slope=sc.aperiodic_slope.reindex(subs).values))
    keep = (np.isfinite(df.fsiq) & np.isfinite(df.iaf) & np.isfinite(df.slope)).values
    df, Z, S, Pe = df[keep].reset_index(drop=True), Z[keep], S[keep], Pe[keep]
    ok = np.isfinite(Z).all(0) & (np.nanstd(Z, 0) > 1e-9)
    feat = [names[k] for k in np.flatnonzero(ok)]
    strength = np.column_stack([stats.rankdata(np.abs(Pe).mean(1)), stats.rankdata(np.abs(Pe).std(1))])
    X = np.column_stack([design(df[['age', 'sex', 'montage', 'iaf', 'slope']]), strength])  # the E1 design
    one = np.ones((len(df), 1))  # residual() ranks, so every block is compared on ranks
    blocks = {'z (adjusted)': residual(X, Z[:, ok]), 'z (unadjusted)': residual(one, Z[:, ok]),
              'channel spectra (adjusted)': residual(X[:, :-2], S), 'Pearson (adjusted)': residual(X, Pe)}
    rank = np.argsort(np.argsort(df.fsiq.values, kind='stable'), kind='stable')
    res, nulls = dict(n=int(len(df)), note='post hoc; not pre-registered'), {}
    for q in FRACTIONS:
        n = int(round(q * len(df)))
        sel = (rank < n) | (rank >= len(df) - n)
        high = rank[sel] >= len(df) - n
        sub = df[sel]
        bal = dict(n_per_group=n, fsiq_low=[float(sub.fsiq[~high].min()), float(sub.fsiq[~high].max())],
                   fsiq_high=[float(sub.fsiq[high].min()), float(sub.fsiq[high].max())])
        for c in ('age', 'iaf', 'slope'):
            bal[c] = [float(sub[c][~high].mean()), float(sub[c][high].mean())]
        for c in ('sex', 'montage', 'education'):
            bal[c] = [sub[c][~high].value_counts().to_dict(), sub[c][high].value_counts().to_dict()]
        r = dict(balance=bal)
        for name, A in blocks.items():
            d, fwer, summ, mx, rms = compare(A[sel], high, rng)
            r[name] = summ; nulls[(q, name)] = (summ, mx, rms)
            if name.startswith('z'):
                top = np.argsort(-np.abs(d))[:10]
                r[name]['top'] = [dict(spi_a=feat[k][0], spi_b=feat[k][1], group=contrast(*feat[k])[0], d=float(d[k]),
                                       fwer=float(fwer[k])) for k in top]
        res[f'{int(q * 100)}%'] = r
    (out / 'extremes.json').write_text(json.dumps(res, indent=1, default=str))
    figure(out, nulls, list(blocks))
    print(json.dumps({k: ({b: {a: x for a, x in v[b].items() if a != 'top'} for b in blocks} if isinstance(v, dict) else v)
                      for k, v in res.items()}, indent=1))


def figure(out, nulls, blocks):
    """Observed statistic against its label-split null, per fraction (columns) and statistic (rows)."""
    style()
    fig, ax = plt.subplots(2, len(FRACTIONS), figsize=(10, 5.2), constrained_layout=True)
    colours = dict(zip(blocks, plt.cm.tab10(np.arange(len(blocks)))))
    for j, q in enumerate(FRACTIONS):
        for i, (key, idx, label) in enumerate((('max_abs_d', 1, r'max$_f\,|d_f|$'), ('rms_d', 2, r'RMS$_f\,d_f$'))):
            a = ax[i, j]
            for b in blocks:
                summ, *nl = nulls[(q, b)]
                v = nl[idx - 1]
                a.hist(v, bins=40, density=True, histtype='step', color=colours[b], lw=1.2, label=b)
                a.axvline(summ[key], color=colours[b], lw=1.7, ls='--')
            a.set(xlabel=label, ylabel='density' if j == 0 else None)
            if i == 0:
                summ = nulls[(q, blocks[0])][0]
                a.set_title(f'top vs bottom {int(q * 100)}% FSIQ ({summ["splits"]} splits)')
    ax[0, -1].legend(loc='upper right', fontsize=7)
    fig.suptitle('Extreme FSIQ groups: observed (dashed) against label-split nulls (outlines); post hoc', fontsize=10)
    save(fig, out, 'iq-extremes'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for a in ('root', 'scope', 'wais', 'out'):
        parser.add_argument(f'--{a}', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.scope, args.wais, args.out)
