"""CHBMP IQ analysis (protocol.md §3). Joins WAIS only with --unblind, after the protocol is frozen.

Subject level: each z feature, core readout and regional lag share is the mean over the subject's windows.
Associations are partial Spearman correlations with FSIQ, adjusted for age, sex, montage, individual alpha frequency and aperiodic slope. Ranks are residualised on the covariates.

- **H1:** posterior minus frontal phase-lagged share, residualised on regional coherence magnitude.
- **H2:** the CORE readouts, Holm-corrected.
- **E1:** every z feature. Family-wise error comes from the max-|ρ| permutation null. Interpretation groups the top 100 by what separates the two SPIs (named axis, within-family parameter, or family pair), with group enrichment against the same permutations.
- **E2:** the top features: edge profiles of low- and high-IQ subjects, leave-one-channel-out stability and correlation with the spectral covariates.
- **H3:** ridge prediction from channel spectra, Pearson and z, as a secondary check.
Sensitivities: education added as a covariate; H1 without Fp1/Fp2 (ocular).
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from mtsspi.character_axes import AXES, core_scores, mt
from mtsspi.figures import save, style
from mtsspi.spi_spi_contract import build_unified_schema
from research.inference.benchmarks.locking_character_axes import family
from research.representation.character_axes.prepare import CANON

FRONTAL = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8']
POSTERIOR = ['P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
ICOH, COH = mt('icoh', 'mean', 'broad'), mt('cohmag', 'mean', 'broad')
TOP, PERMS = 100, 10000
AXIS_OF = {frozenset(p): ax.name for ax in AXES for p in ax.pairs}
# SPIs that are functions of the auto- and cross-spectral matrix: a pair of two of them is linear-spectral by
# construction and cannot read non-Gaussian form (axes-review reply, 5 November).
SECOND_ORDER = ('cov', 'cov-sq', 'prec', 'prec-sq', 'xcorr', 'xcorr-sq', 'cohmag', 'icoh', 'phase', 'gd', 'psi', 'sgc',
                'dtf', 'ddtf', 'dcoh', 'pdcoh', 'gpdcoh', 'mi_gaussian', 'tlmi_gaussian', 'je_gaussian', 'gc_gaussian',
                'pdist_euclidean', 'xpdist_euclidean', 'corr_pearson')
# Dominant nuisance measured in the construction validation, where known.
NUISANCE = {'linear and rank correlation: lag': 'coupling frequency', 'axis: timing_fixed_lag': 'coupling frequency',
            'axis: form_monotone': 'marginal shape', 'axis: tail': 'marginal shape', 'axis: band': 'within-band location',
            'axis: timing_warping': 'insensitive at cohort coupling', 'axis: form_information': 'insensitive at cohort coupling'}


def second_order(name):
    return name.startswith(SECOND_ORDER)


def contrast(a, b):
    """What separates two SPIs, and whether that is a single assumption.

    A named minimal-pair axis or a within-family difference in one parameter kind isolates one assumption. Pairs
    from different estimator families are grouped by family pair and marked compound, as are within-family pairs
    that differ in several parameter kinds."""
    if frozenset((a, b)) in AXIS_OF:
        return f'axis: {AXIS_OF[frozenset((a, b))]}', True
    fa, fb = family(a), family(b)
    if fa != fb:
        return ' × '.join(sorted((fa, fb))), False
    diff = ' '.join(sorted(set(a.split('_')) ^ set(b.split('_'))))
    kinds = {label for key, label in (('tau-', 'lag'), ('fmin', 'band'), ('fmax', 'band'), ('k-', 'history'),
                                      ('-sq', 'squared'), ('mean', 'spectral aggregation'), ('max', 'spectral aggregation'))
             if key in diff}
    if len(kinds) == 1:
        return f'{fa}: {kinds.pop()}', True
    return (f'{fa}: compound', False) if kinds else (f'{fa}: estimator', True)


def load(root, scope):
    with np.load(root / 'bank.npz', allow_pickle=False) as b:
        bank = {k: b[k] for k in ('z', 'aligned', 'group', 'row_id', 'spi_order', 'M', 'pearson')}
    rows = json.loads((root / 'manifest.json').read_text())['rows']
    meta = pd.DataFrame(rows).set_index('row_id').loc[bank['row_id']]
    with np.load(root / 'absolute-power.npz') as p:
        power = pd.DataFrame(p['power'], index=p['row_id']).loc[bank['row_id']].values
    sc = pd.read_csv(scope / 'records.csv').set_index('subject')
    return bank, meta, power, sc


def profiles(bank):
    order = list(bank['spi_order']); M = int(bank['M'][0]); E = M * (M - 1)
    return bank['aligned'].reshape(len(bank['aligned']), len(order), E), order, M


def edge_index(M):
    i, j = np.where(~np.eye(M, dtype=bool))
    return i, j


def regional_share(prof, order, M, drop=()):
    i, j = edge_index(M)
    ic, co = prof[:, order.index(ICOH)], prof[:, order.index(COH)]
    with np.errstate(invalid='ignore', divide='ignore'):
        share = ic / np.sqrt(co); mag = np.sqrt(co)
    out = {}
    for name, region in (('frontal', FRONTAL), ('posterior', POSTERIOR)):
        k = [CANON.index(c) for c in region if c not in drop]
        e = np.isin(i, k) & np.isin(j, k)
        out[name] = np.nanmean(share[:, e], 1); out[name + '_mag'] = np.nanmean(mag[:, e], 1)
    return out


def subject_mean(x, group):
    subs = np.unique(group)
    return subs, np.stack([np.nanmean(x[group == s], 0) for s in subs])


def design(cov):
    """Ranked continuous covariates plus categorical dummies, with intercept."""
    cols = [np.ones(len(cov))]
    for c in cov.columns:
        v = cov[c]
        if not pd.api.types.is_numeric_dtype(v):
            cols += [(v == u).astype(float).values for u in sorted(v.dropna().unique())[1:]]
        else:
            cols.append(stats.rankdata(v))
    return np.column_stack(cols).astype(float)


def residual(X, A):
    """Residuals of the ranked columns of A on design X."""
    R = np.apply_along_axis(stats.rankdata, 0, np.atleast_2d(A.T).T)
    beta, *_ = np.linalg.lstsq(X, R, rcond=None)
    return R - X @ beta


def partial_spearman(X, A, y, perms=PERMS, rng=None):
    """Partial Spearman rho of each column of A with y, with max-|rho| permutation p-values (FWER) and per-feature p."""
    rng = rng or np.random.default_rng(261105)
    ra, ry = residual(X, A), residual(X, y)[:, 0]
    sa = (ra - ra.mean(0)) / ra.std(0); sy = (ry - ry.mean()) / ry.std()
    rho = sa.T @ sy / len(sy)
    P = np.stack([rng.permutation(sy) for _ in range(perms)], 1)
    null = sa.T @ P / len(sy)
    per = (1 + (np.abs(null) >= np.abs(rho)[:, None]).sum(1)) / (perms + 1)
    fwer = (1 + (np.abs(null).max(0)[None, :] >= np.abs(rho)[:, None]).sum(1)) / (perms + 1)
    return rho, per, fwer, null


def holm(p):
    order = np.argsort(p); out = np.empty_like(p); m = len(p); run = 0
    for r, k in enumerate(order):
        run = max(run, (m - r) * p[k]); out[k] = min(1, run)
    return out


def main(root, scope, wais, frozen, out):
    out.mkdir(parents=True, exist_ok=True)
    bank, meta, power, sc = load(root, scope)
    prof, order, M = profiles(bank)
    group = bank['group']
    subs, Z = subject_mean(bank['z'].astype(float), group)
    names = [(f.spi_a, f.spi_b) for f in build_unified_schema(order)]
    core, core_names = core_scores(prof.astype(np.float64), order)
    _, C = subject_mean(core, group)
    reg = regional_share(prof, order, M); reg_nofp = regional_share(prof, order, M, drop=('Fp1', 'Fp2'))
    R = {k: subject_mean(v, group)[1] for k, v in reg.items()}; Rn = {k: subject_mean(v, group)[1] for k, v in reg_nofp.items()}
    _, S = subject_mean(power, group); _, Pe = subject_mean(bank['pearson'].astype(float), group)
    first = meta.groupby('group').first().loc[subs]
    w = pd.read_csv(wais, skiprows=1, encoding='utf-8-sig')
    w.columns = [c.strip() for c in w.columns]
    fsiq = w.assign(subject='sub-' + w.Code.astype(str).str.strip()).set_index('subject')['FSIQ']
    df = pd.DataFrame(dict(subject=subs, fsiq=pd.to_numeric(fsiq.reindex(subs), errors='coerce').values,
                           age=first.age.values, sex=first.sex.values, montage=first.montage.astype(str).values,
                           education=first.education.values,
                           iaf=sc.iaf.reindex(subs).values, slope=sc.aperiodic_slope.reindex(subs).values))
    keep = np.isfinite(df.fsiq) & np.isfinite(df.iaf) & np.isfinite(df.slope)
    df, Z, C, S, Pe = df[keep].reset_index(drop=True), Z[keep], C[keep], S[keep], Pe[keep]
    R = {k: v[keep] for k, v in R.items()}; Rn = {k: v[keep] for k, v in Rn.items()}
    X = design(df[['age', 'sex', 'montage', 'iaf', 'slope']]); Xe = design(df[['age', 'sex', 'montage', 'iaf', 'slope', 'education']])
    strength = np.column_stack([stats.rankdata(np.abs(Pe).mean(1)), stats.rankdata(np.abs(Pe).std(1))])
    Xs = np.column_stack([X, strength])
    sp = np.linalg.svd((S - S.mean(0)) / S.std(0), full_matrices=False)[0][:, :5]
    Xsp = np.column_stack([Xs, sp])
    y = df.fsiq.values
    res = dict(frozen_commit=frozen, n=int(len(df)), fsiq=dict(mean=float(y.mean()), sd=float(y.std()), min=float(y.min()), max=float(y.max())))

    # H1
    def h1(Rr, Xd):
        D = Rr['posterior'] - Rr['frontal']
        Xh = np.column_stack([Xd, stats.rankdata(Rr['frontal_mag']), stats.rankdata(Rr['posterior_mag'])])
        rho, p, _, _ = partial_spearman(Xh, D[:, None], y)
        parts = {k: float(partial_spearman(Xh, Rr[k][:, None], y, perms=2000)[0][0]) for k in ('frontal', 'posterior')}
        return dict(rho=float(rho[0]), p=float(p[0]), parts=parts), (residual(Xh, D)[:, 0], residual(Xh, y)[:, 0])
    res['H1'], D = h1(R, X)
    res['H1_education'] = h1(R, Xe)[0]; res['H1_without_Fp'] = h1(Rn, X)[0]

    # H2
    rho, p, _, _ = partial_spearman(X, C, y)
    res['H2'] = {n: dict(rho=float(r), p=float(q), p_holm=float(h)) for n, r, q, h in zip(core_names, rho, p, holm(p))}

    # E1
    ok = np.isfinite(Z).all(0) & (np.nanstd(Z, 0) > 1e-9)
    rho, p, fwer, null = partial_spearman(Xs, Z[:, ok], y)
    feat = [names[k] for k in np.flatnonzero(ok)]
    lab = [contrast(a, b) for a, b in feat]
    labels = np.array([g for g, _ in lab]); isolates = np.array([i for _, i in lab])
    linear_spectral = np.array([second_order(a) and second_order(b) for a, b in feat])
    top = np.argsort(-np.abs(rho))[:TOP]
    # Distinct leading features: walk down the ranking, skipping any whose subject-level z correlates |r| > 0.9
    # with one already kept, so near-duplicates (one finding under several estimators) are shown once.
    Zr = residual(Xs, Z[:, ok][:, top]); Zr = (Zr - Zr.mean(0)) / Zr.std(0)
    distinct = []
    for n in range(TOP):
        if all(abs(Zr[:, n] @ Zr[:, m]) / len(Zr) <= .9 for m in distinct):
            distinct.append(n)
    distinct = top[distinct]
    obs = Counter(labels[top]); base = Counter(labels)
    groups = [g for g, _ in obs.most_common(15)]
    perm_counts = []
    for k in range(0, null.shape[1], 10):
        idx = np.argpartition(-np.abs(null[:, k]), TOP)[:TOP]
        perm_counts.append([np.sum(labels[idx] == g) for g in groups])
    perm_counts = np.array(perm_counts)
    res['E1'] = dict(features=int(ok.sum()), fwer_significant=int((fwer < .05).sum()), min_fwer=float(fwer.min()),
                     max_abs_rho=float(np.abs(rho).max()), null_max_abs_rho_95=float(np.quantile(np.abs(null).max(0), .95)),
                     covariates='age, sex, montage, IAF, aperiodic slope, mean and SD of |Pearson|',
                     top_isolating_share=float(isolates[top].mean()), top_linear_spectral_share=float(linear_spectral[top].mean()),
                     catalogue_linear_spectral_share=float(linear_spectral.mean()),
                     distinct_top=[dict(spi_a=feat[k][0], spi_b=feat[k][1], group=labels[k], rho=float(rho[k]), fwer=float(fwer[k])) for k in distinct[:12]],
                     top=[dict(spi_a=feat[k][0], spi_b=feat[k][1], group=labels[k], isolates=bool(isolates[k]),
                               linear_spectral=bool(linear_spectral[k]), known_nuisance=NUISANCE.get(labels[k]),
                               rho=float(rho[k]), p=float(p[k]), fwer=float(fwer[k])) for k in top[:25]],
                     groups=[dict(group=g, known_nuisance=NUISANCE.get(g), top_count=int(obs[g]), expected=float(TOP * base[g] / len(labels)),
                                  p_enrichment=float((1 + (perm_counts[:, i] >= obs[g]).sum()) / (1 + len(perm_counts))))
                             for i, g in enumerate(groups)])
    rho_e = partial_spearman(np.column_stack([Xe, strength]), Z[:, ok][:, top[:25]], y, perms=2000)[0]
    rho_s, p_s, _, _ = partial_spearman(Xsp, Z[:, ok][:, top[:25]], y, perms=2000)
    for t, r, r2, q2 in zip(res['E1']['top'], rho_e, rho_s, p_s):
        t.update(rho_with_education=float(r), rho_with_spectra=float(r2), p_with_spectra=float(q2))

    # E2: leave-one-channel-out for the 5 leading distinct features and correlation with the spectral covariates
    sub_rows = {s: np.flatnonzero(group == s) for s in df.subject}
    i, j = edge_index(M)
    e2 = []
    for k in distinct[:5]:
        a, b = feat[k]; ia, ib = order.index(a), order.index(b)
        loo = []
        for c in range(M):
            e = (i != c) & (j != c)
            zc = np.array([np.nanmean([np.corrcoef(prof[r, ia, e], prof[r, ib, e])[0, 1] for r in sub_rows[s]]) for s in df.subject])
            loo.append(float(partial_spearman(Xs, zc[:, None], y, perms=1)[0][0]))
        zf = Z[:, ok][:, k]
        e2.append(dict(spi_a=a, spi_b=b, rho=float(rho[k]), loo_min=min(loo, key=abs), loo_sign_stable=bool(np.all(np.sign(loo) == np.sign(rho[k]))),
                       loo_by_channel=dict(zip(CANON if M == 19 else range(M), loo)),
                       r_with_iaf=float(stats.spearmanr(zf, df.iaf)[0]), r_with_slope=float(stats.spearmanr(zf, df.slope)[0])))
    res['E2'] = e2

    # H3
    res['H3'] = ridge(dict(spectra=S, pearson=Pe, z=Z[:, ok]), y, df)
    (out / 'results.json').write_text(json.dumps(res, indent=1))
    shown = distinct[:4]
    adjusted = (residual(Xs, Z[:, ok][:, shown]), residual(Xs, y)[:, 0])
    figures(out, df, rho, p, fwer, labels, shown, feat, prof, order, group, D, res, Z[:, ok], adjusted)
    print(json.dumps({k: res[k] for k in ('n', 'H1', 'H1_education', 'H1_without_Fp')}, indent=1))
    print(json.dumps(res['E1']['groups'][:10], indent=1))


def ridge(blocks, y, df, repeats=5, folds=10):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import RepeatedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cov = np.column_stack([df.age, (df.sex == 'M').astype(float), (df.montage == df.montage.iloc[0]).astype(float), df.iaf, df.slope])
    sets = dict(covariates=cov, spectra=np.column_stack([cov, blocks['spectra']]),
                spectra_pearson=np.column_stack([cov, blocks['spectra'], blocks['pearson']]))
    out = {}
    for name, A in list(sets.items()) + [('spectra_pearson_z', None)]:
        pred = np.zeros((repeats, len(y)))
        for r, (tr, te) in enumerate(RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=261105).split(y)):
            rep = r // folds
            if A is None:
                pca = make_pipeline(StandardScaler(), PCA(min(50, len(tr) - 1))).fit(blocks['z'][tr])
                Atr = np.column_stack([sets['spectra_pearson'][tr], pca.transform(blocks['z'][tr])])
                Ate = np.column_stack([sets['spectra_pearson'][te], pca.transform(blocks['z'][te])])
            else:
                Atr, Ate = A[tr], A[te]
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 30))).fit(Atr, y[tr])
            pred[rep, te] = m.predict(Ate)
        r2 = [1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2) for p in pred]
        out[name] = dict(r2_mean=float(np.mean(r2)), r2_sd=float(np.std(r2)))
    return out


def figures(out, df, rho, p, fwer, labels, shown, feat, prof, order, group, D, res, Zok, adjusted):
    """Scatters show covariate-adjusted ranks, the quantities whose correlation is the reported partial rho."""
    style()
    groups = [g['group'] for g in res['E1']['groups'][:8]]
    colours = dict(zip(groups, plt.cm.tab10(np.arange(len(groups)))))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.2, 1]))
    a = ax[0]
    other = ~np.isin(labels, groups)
    a.scatter(rho[other], -np.log10(p[other]), s=2, color='.8', rasterized=True, label='other groups')
    for g in groups:
        k = labels == g
        a.scatter(rho[k], -np.log10(p[k]), s=3, color=colours[g], rasterized=True, label=g)
    thr = res['E1']['null_max_abs_rho_95']
    for s in (-1, 1):
        a.axvline(s * thr, color='k', ls=':', lw=1)
    a.set(xlabel=r'partial Spearman $\rho$ (z feature, FSIQ)', ylabel=r'$-\log_{10}$ permutation p', title='Every SPI–SPI feature against FSIQ (dotted: FWER 5%)')
    a.legend(fontsize=6, markerscale=3, loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2, frameon=False)
    a = ax[1]
    g = res['E1']['groups'][:12]
    y = np.arange(len(g))[::-1]
    a.barh(y, [x['top_count'] for x in g], color=[colours.get(x['group'], '.6') for x in g])
    a.scatter([x['expected'] for x in g], y, color='k', marker='|', s=80, label='expected from catalogue')
    for yy, x in zip(y, g):
        if x['p_enrichment'] < .05:
            a.text(x['top_count'] + .3, yy, '*', va='center')
    a.set_yticks(y, [x['group'] for x in g], fontsize=7)
    a.set(xlabel=f'features among the top {TOP}', title='What the top features contrast (* enriched, p < .05)')
    a.legend()
    save(fig, out, 'iq-features'); plt.close(fig)

    lo, hi = df.fsiq <= df.fsiq.quantile(1 / 3), df.fsiq >= df.fsiq.quantile(2 / 3)
    fig, ax = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    fig.suptitle('Leading distinct features (|r| ≤ 0.9 between them). Top: subject-mean edge profiles; bottom: adjusted ranks', fontsize=9)
    for c, k in enumerate(shown):
        a_, b_ = feat[k]; ia, ib = order.index(a_), order.index(b_)
        a = ax[0, c]
        for mask, col, lab in ((lo, '#4C72B0', 'low-IQ tertile'), (hi, '#C44E52', 'high-IQ tertile')):
            rows = np.isin(group, df.subject[mask])
            xa, xb = np.nanmean(prof[rows, ia], 0), np.nanmean(prof[rows, ib], 0)
            a.scatter(xa, xb, s=4, color=col, alpha=.7, label=lab)
        a.set(xlabel=short(a_), ylabel=short(b_), title=f'{labels[k]}', )
        a.title.set_fontsize(7)
        if c == 0:
            a.legend(fontsize=6)
        a = ax[1, c]
        a.scatter(adjusted[1], adjusted[0][:, c], s=6, color='k')
        a.set(xlabel='FSIQ rank, adjusted', ylabel='z rank, adjusted', title=fr'$\rho_p$ = {rho[k]:.2f}, FWER p = {fwer[k]:.3f}')
    save(fig, out, 'iq-top-features'); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.4), constrained_layout=True)
    ax[0].scatter(D[1], D[0], s=8, color='k')
    ax[0].set(xlabel='FSIQ rank, adjusted', ylabel='posterior − frontal lagged share, rank adjusted', title=f"H1: ρ_p = {res['H1']['rho']:.2f}, p = {res['H1']['p']:.3f}")
    from sklearn.decomposition import PCA
    pc = PCA(2).fit((Zok - Zok.mean(0)) / Zok.std(0))
    e = pc.transform((Zok - Zok.mean(0)) / Zok.std(0))
    s = ax[1].scatter(e[:, 0], e[:, 1], c=df.fsiq, cmap='viridis', s=10)
    plt.colorbar(s, ax=ax[1], label='FSIQ')
    ax[1].set(xlabel=f'PC1 ({pc.explained_variance_ratio_[0]:.0%})', ylabel=f'PC2 ({pc.explained_variance_ratio_[1]:.0%})', title='Subject z, PCA')
    ax[1].set_aspect('equal', adjustable='datalim')
    save(fig, out, 'iq-h1-embedding'); plt.close(fig)


def short(name):
    return re.sub(r'_multitaper|_fs-1', '', name).replace('EmpiricalCovariance', 'empirical')[:40]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='prepared and computed corpus root (bank.npz)')
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--wais', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--unblind', action='store_true')
    parser.add_argument('--frozen-commit', help='commit at which protocol.md and this script were frozen')
    args = parser.parse_args()
    if not (args.unblind and args.frozen_commit):
        parser.error('WAIS is joined only with --unblind and --frozen-commit (protocol.md §5)')
    main(args.root, args.scope, args.wais, args.frozen_commit, args.out)
