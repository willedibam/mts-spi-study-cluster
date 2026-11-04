"""Replication in LEMON (old against young) of the CHBMP z-age association; targets are fixed by CHBMP alone (results.md).

Discovery recomputes the CHBMP age designs of controls.py (same seed, so the 235 FWER features are the same). LEMON uses the
matched design: sex, IAF, aperiodic slope and Pearson strength, then + 5 spectral PCs of absolute band power.
- T1: sign concordance, mean of sign(rho_CHBMP) * rho_LEMON over the CHBMP targets (age_full: FWER < .05; spectra
  design: its top 100), one-sided against LEMON label permutations, which keep the dependence between features.
- T2: Pearson r between the two cohorts' rho over all common features, against the same permutations.
- T3: the composite of directed VAR spectral (dcoh, gpdcoh) x undirected (cohmag, PLV, PPC) features, mean of
  subject-standardised z; predicted rho > 0 with age.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

from mtsspi.figures import save, style
from mtsspi.spi_spi_contract import build_unified_schema
from research.representation.chbmp_iq.analysis import design, load, partial_spearman, residual, subject_mean
from research.representation.character_axes.prepare import CANON

DIRECTED, UNDIRECTED = ('dcoh_', 'gpdcoh_'), ('cohmag_', 'plv_', 'ppc_')
PERMS = 5000


def strength_of(Pe):
    return np.column_stack([stats.rankdata(np.abs(Pe).mean(1)), stats.rankdata(np.abs(Pe).std(1))])


def spectral_pcs(S):
    return np.linalg.svd((S - S.mean(0)) / S.std(0), full_matrices=False)[0][:, :5]


def chbmp(root, scope):
    """CHBMP age rho for every feature under the controls.py age_full and age_full_spectra designs."""
    bank, meta, power, sc = load(root, scope)
    group = bank['group']
    subs, Z = subject_mean(bank['z'].astype(float), group)
    _, S = subject_mean(power, group); _, Pe = subject_mean(bank['pearson'].astype(float), group)
    first = meta.groupby('group').first().loc[subs]
    cov = pd.DataFrame(dict(age=first.age.values, sex=first.sex.values, montage=first.montage.astype(str).values,
                            iaf=sc.iaf.reindex(subs).values, slope=sc.aperiodic_slope.reindex(subs).values))
    ok = np.isfinite(Z).all(0) & (np.nanstd(Z, 0) > 1e-9)
    keep = cov[['sex', 'montage', 'iaf', 'slope']].notna().all(1).values
    X = np.column_stack([design(cov[keep][['sex', 'montage', 'iaf', 'slope']]), strength_of(Pe)[keep]])
    out = {}
    for name, Xa in (('full', X), ('spectra', np.column_stack([X, spectral_pcs(S)[keep]]))):
        rho, _, fwer, _ = partial_spearman(Xa, Z[keep][:, ok], cov.age.values[keep], perms=2000)
        out[name] = (rho, fwer)
    return ok, out, bank['spi_order']


def lemon(root):
    with np.load(root / 'bank.npz', allow_pickle=False) as b:
        bank = {k: b[k] for k in ('z', 'group', 'row_id', 'pearson', 'x', 'spi_order')}
    rows = json.loads((root / 'manifest.json').read_text())['rows']
    meta = pd.DataFrame(rows).set_index('row_id').loc[bank['row_id']]
    with np.load(root / 'absolute-power.npz') as p:
        power = pd.DataFrame(p['power'], index=p['row_id']).loc[bank['row_id']].values
    group = bank['group']
    subs, Z = subject_mean(bank['z'].astype(float), group)
    _, S = subject_mean(power, group); _, Pe = subject_mean(bank['pearson'].astype(float), group)
    first = meta.groupby('group').first().loc[subs]
    # IAF and aperiodic slope as in chbmp_iq/scope.py: occipital mean spectrum, 7-13 Hz peak, 2-40 Hz fit without 7-14 Hz
    occ = [CANON.index(c) for c in ('O1', 'O2', 'Pz')]
    f, p = signal.welch(bank['x'][:, :, occ].astype(float), fs=100, nperseg=200, axis=1)
    po = np.stack([p[group == s].mean((0, 2)) for s in subs])
    k, kk = (f >= 7) & (f <= 13), (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    cov = pd.DataFrame(dict(sex=first.sex.astype(str).values, iaf=f[k][po[:, k].argmax(1)],
                            slope=[np.polyfit(np.log10(f[kk]), np.log10(q[kk]), 1)[0] for q in po]))
    old = first.target.values.astype(float)
    return subs, Z, S, Pe, cov, old, first.age_bin.values, bank['spi_order']


def main(chbmp_root, scope, lemon_root, out):
    out.mkdir(parents=True, exist_ok=True)
    okC, disc, orderC = chbmp(chbmp_root, scope)
    subs, Z, S, Pe, cov, old, bins, orderL = lemon(lemon_root)
    if not np.array_equal(orderC, orderL):
        raise ValueError('SPI orders differ between the banks')
    names = [(f.spi_a, f.spi_b) for f in build_unified_schema(orderC)]
    okL = np.isfinite(Z).all(0) & (np.nanstd(Z, 0) > 1e-9)
    common = okC & okL
    inC = common[okC]  # common features within CHBMP's ok set
    feat = [names[k] for k in np.flatnonzero(common)]
    X = np.column_stack([design(cov[['sex', 'iaf', 'slope']]), strength_of(Pe)])
    designs = dict(full=X, spectra=np.column_stack([X, spectral_pcs(S)]))
    comp = np.array([(a.startswith(DIRECTED) and b.startswith(UNDIRECTED)) or (b.startswith(DIRECTED) and a.startswith(UNDIRECTED))
                     for a, b in feat])
    res = dict(lemon_n=int(len(subs)), old=int(old.sum()), young=int((1 - old).sum()), age_bins=pd.Series(bins).value_counts().to_dict(),
               common_features=int(common.sum()), chbmp_full_fwer_count=int((disc['full'][1] < .05).sum()),
               composite_features=int(comp.sum()), iaf=[float(cov.iaf[old == g].mean()) for g in (0, 1)],
               slope=[float(cov.slope[old == g].mean()) for g in (0, 1)])
    Zs = (Z[:, common] - Z[:, common].mean(0)) / Z[:, common].std(0)
    composite = Zs[:, comp].mean(1)
    shown = {}
    for name, Xd in designs.items():
        rhoC, fwerC = (v[inC] for v in disc[name])
        rho, _, fwer, null = partial_spearman(Xd, Z[:, common], old, perms=PERMS)
        targets = fwerC < .05 if name == 'full' else np.isin(np.arange(len(rhoC)), np.argsort(-np.abs(rhoC))[:100])
        sgn = np.sign(rhoC[targets])
        conc, cnull = np.mean(sgn * rho[targets]), (sgn[:, None] * null[targets]).mean(0)
        zc = (rhoC - rhoC.mean()) / rhoC.std()
        nz = (null - null.mean(0)) / null.std(0)
        prof, pnull = np.mean(zc * (rho - rho.mean()) / rho.std()), zc @ nz / len(zc)
        crho, cp, _, cnl = partial_spearman(Xd, composite[:, None], old, perms=10000)
        res[name] = dict(
            T1=dict(targets=int(targets.sum()), concordance=float(conc), p_one_sided=float((1 + (cnull >= conc).sum()) / (PERMS + 1)),
                    same_sign_share=float(np.mean(sgn == np.sign(rho[targets])))),
            T2=dict(r=float(prof), p_one_sided=float((1 + (pnull >= prof).sum()) / (PERMS + 1))),
            T3=dict(rho=float(crho[0]), p_two_sided=float(cp[0]), p_one_sided=float((1 + (cnl[0] >= crho[0]).sum()) / 10001)),
            lemon_fwer_significant=int((fwer < .05).sum()), lemon_max_abs_rho=float(np.abs(rho).max()),
            lemon_null_max_abs_rho_95=float(np.quantile(np.abs(null).max(0), .95)),
            lemon_top=[dict(spi_a=feat[k][0], spi_b=feat[k][1], rho=float(rho[k]), fwer=float(fwer[k]), chbmp_rho=float(rhoC[k]))
                       for k in np.argsort(-np.abs(rho))[:15]])
        shown[name] = (rhoC, rho, targets, residual(Xd, composite)[:, 0])
    (out / 'lemon-age.json').write_text(json.dumps(res, indent=1, default=str))
    figure(out, res, shown, comp, old)
    print(json.dumps({k: ({a: b for a, b in v.items() if a != 'lemon_top'} if isinstance(v, dict) else v) for k, v in res.items()}, indent=1))


def figure(out, res, shown, comp, old):
    style()
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
    for a, name, title in ((ax[0], 'full', 'sex, IAF, slope, strength'), (ax[1], 'spectra', '+ 5 spectral PCs')):
        rhoC, rho, targets, _ = shown[name]
        a.scatter(rhoC, rho, s=1.5, color='.75', rasterized=True, label='all common features')
        a.scatter(rhoC[comp], rho[comp], s=3, color='tab:orange', rasterized=True, label='directed × undirected spectral')
        a.scatter(rhoC[targets], rho[targets], s=5, facecolor='none', edgecolor='tab:blue', lw=.5, label='CHBMP targets')
        a.axhline(0, color='k', lw=.6); a.axvline(0, color='k', lw=.6)
        r = res[name]
        a.set(xlabel=r'CHBMP partial $\rho$ (age, continuous)', ylabel=r'LEMON partial $\rho$ (old vs young)',
              title=f'{title}\nT1 {r["T1"]["concordance"]:+.3f} (p {r["T1"]["p_one_sided"]:.3g}), T2 r {r["T2"]["r"]:+.2f} (p {r["T2"]["p_one_sided"]:.3g})')
        a.set_box_aspect(1)
    ax[0].legend(loc='upper left', fontsize=6, markerscale=3)
    a = ax[2]
    for i, name in enumerate(('full', 'spectra')):
        c = shown[name][3]
        for g, colour in ((0, 'tab:green'), (1, 'tab:purple')):
            v = c[old == g]
            a.scatter(i * 2.4 + g + np.random.default_rng(g).uniform(-.18, .18, len(v)), v, s=4, color=colour, alpha=.7)
            a.plot([i * 2.4 + g - .25, i * 2.4 + g + .25], [np.median(v)] * 2, color='k', lw=1.7)
    a.set_xticks([0, 1, 2.4, 3.4], ['young', 'old', 'young', 'old'])
    a.set(ylabel='composite, residual rank', title=f'T3 composite: $\\rho$ {res["full"]["T3"]["rho"]:+.2f} / {res["spectra"]["T3"]["rho"]:+.2f} (+PCs)')
    a.text(.5, -.2, 'base design', ha='center', transform=a.get_xaxis_transform(), fontsize=8)
    a.text(2.9, -.2, '+ spectral PCs', ha='center', transform=a.get_xaxis_transform(), fontsize=8)
    save(fig, out, 'lemon-age'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for a in ('chbmp_root', 'scope', 'lemon_root', 'out'):
        parser.add_argument(f'--{a.replace("_", "-")}', type=Path, required=True)
    args = parser.parse_args()
    main(args.chbmp_root, args.scope, args.lemon_root, args.out)
