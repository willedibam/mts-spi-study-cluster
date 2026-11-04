"""Post-hoc controls for the frozen CHBMP IQ analysis (not pre-registered; protocol.md is unchanged).

1. location_frequency: which subjects are undefined, and its FSIQ association on complete cases.
2. Cohort reliability of subject-mean z: odd/even-window split half (Spearman-Brown) and identification.
3. Positive control: every z feature against age on all prepared subjects, max-|rho| FWER, top-100 group counts.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mtsspi.character_axes import core_scores
from mtsspi.spi_spi_contract import build_unified_schema
from research.representation.chbmp_iq.analysis import (TOP, contrast, design, load, partial_spearman, profiles,
                                                       subject_mean)


def split_half(z, group, subs):
    """Subject means over odd and even windows; per-feature Spearman-Brown reliability and identification rate."""
    halves = [[], []]
    for s in subs:
        x = z[group == s]
        halves[0].append(np.nanmean(x[0::2], 0)); halves[1].append(np.nanmean(x[1::2], 0))
    a, b = (np.array(h) for h in halves)
    fin = np.isfinite(a).all(0) & np.isfinite(b).all(0)  # a half can have no finite window for a feature
    a, b = a[:, fin], b[:, fin]
    a, b = (a - a.mean(0)) / a.std(0), (b - b.mean(0)) / b.std(0)
    r = (a * b).mean(0)
    rows = lambda x: (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    hit = np.argmax(rows(a) @ rows(b).T, 1) == np.arange(len(a))
    return np.median(2 * r / (1 + r)), float(np.mean(r > .5)), float(hit.mean()), int(fin.sum())


def main(root, scope, wais, out):
    out.mkdir(parents=True, exist_ok=True)
    bank, meta, power, sc = load(root, scope)
    prof, order, M = profiles(bank)
    group = bank['group']; z = bank['z'].astype(float)
    subs, Z = subject_mean(z, group)
    names = [(f.spi_a, f.spi_b) for f in build_unified_schema(order)]
    first = meta.groupby('group').first().loc[subs]
    res = {}

    core, core_names = core_scores(prof.astype(np.float64), order)
    loc = core[:, core_names.index('location_frequency')]
    per = np.array([np.isnan(loc[group == s]).mean() for s in subs])
    w = pd.read_csv(wais, skiprows=1, encoding='utf-8-sig'); w.columns = [c.strip() for c in w.columns]
    w = w.dropna(subset=['Code'])
    fsiq = pd.to_numeric(w.assign(subject='sub-' + w.Code.astype(str).str.strip()).set_index('subject').FSIQ,
                         errors='coerce').reindex(subs).values
    _, Pe = subject_mean(bank['pearson'].astype(float), group)
    strength = np.column_stack([stats.rankdata(np.abs(Pe).mean(1)), stats.rankdata(np.abs(Pe).std(1))])
    cov = pd.DataFrame(dict(age=first.age.values, sex=first.sex.values, montage=first.montage.astype(str).values,
                            iaf=sc.iaf.reindex(subs).values, slope=sc.aperiodic_slope.reindex(subs).values))
    _, L = subject_mean(loc[:, None], group)
    cc = np.isfinite(fsiq) & np.isfinite(L[:, 0]) & cov[['iaf', 'slope']].notna().all(1).values
    X = np.column_stack([design(cov[cc]), strength[cc]])
    rho, p, _, _ = partial_spearman(X, L[cc], fsiq[cc], perms=10000)
    res['location_frequency'] = dict(nan_windows=int(np.isnan(loc).sum()), windows=len(loc),
                                     subjects_all_nan=[str(s) for s in subs[per == 1]],
                                     subjects_any_nan=int((per > 0).sum()), complete_case_n=int(cc.sum()),
                                     complete_case_rho=float(rho[0]), complete_case_p=float(p[0]))

    ok = np.isfinite(Z).all(0) & (np.nanstd(Z, 0) > 1e-9)
    sb, share, ident, used = split_half(z[:, ok], group, subs)
    res['reliability'] = dict(subjects=len(subs), features=used, split_half_spearman_brown_median=float(sb),
                              share_half_r_over_half=share, identification_odd_to_even=ident)

    feat = [names[k] for k in np.flatnonzero(ok)]
    labels = np.array([contrast(a, b)[0] for a, b in feat])
    _, S = subject_mean(power, group)
    sp = np.linalg.svd((S - S.mean(0)) / S.std(0), full_matrices=False)[0][:, :5]
    for name, cols in (('age_sex_montage', ['sex', 'montage']),
                       ('age_full', ['sex', 'montage', 'iaf', 'slope']),
                       ('age_full_spectra', ['sex', 'montage', 'iaf', 'slope'])):
        keep = cov[cols].notna().all(1).values
        Xa = design(cov[keep][cols])
        extra = {'age_full': [strength], 'age_full_spectra': [strength, sp]}.get(name, [])
        Xa = np.column_stack([Xa] + [e[keep] for e in extra])
        rho, p, fwer, null = partial_spearman(Xa, Z[keep][:, ok], cov.age.values[keep], perms=2000)
        top = np.argsort(-np.abs(rho))[:TOP]
        obs, base = Counter(labels[top]), Counter(labels)
        groups = [g for g, _ in obs.most_common(8)]
        counts = np.array([[np.sum(labels[np.argpartition(-np.abs(null[:, k]), TOP)[:TOP]] == g) for g in groups]
                           for k in range(0, null.shape[1], 4)])
        res[name] = dict(n=int(keep.sum()), covariates=cols + {'age_full': ['strength'], 'age_full_spectra': ['strength', '5 spectral PCs']}.get(name, []),
                         fwer_significant=int((fwer < .05).sum()), max_abs_rho=float(np.abs(rho).max()),
                         null_max_abs_rho_95=float(np.quantile(np.abs(null).max(0), .95)),
                         groups=[dict(group=g, top_count=obs[g], expected=TOP * base[g] / len(labels),
                                      p_enrichment=float((1 + (counts[:, i] >= obs[g]).sum()) / (len(counts) + 1)))
                                 for i, g in enumerate(groups)],
                         top=[dict(spi_a=feat[k][0], spi_b=feat[k][1], group=labels[k], rho=float(rho[k]), fwer=float(fwer[k]))
                              for k in top[:15]])
    (out / 'controls.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'top'} for k, v in res.items()}, indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--wais', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.scope, args.wais, args.out)
