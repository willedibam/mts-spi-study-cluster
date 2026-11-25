"""Score the surrogate-ensemble fourth-order readouts of kens.py (rule ../preregs/kens-prereg.md and its amendment).

Written and committed before any ke_*.npz output was read. Choices the prereg leaves open, fixed here at that commit:
- Reference arm 'im' (normal-scored IAAFT draws) for every readout; 'ia' (spectrum-final IAAFT) and 'ft' (phase-randomised) are
  diagnostics. z = (real - arm mean) / arm sd per pair; pairs with arm sd = 0 are counted and printed, then dropped from the mean.
- Tail fraction: share of pairs whose real value exceeds at least B - 2 of the B 'im' draws (chance about 3 / (B + 1)).
- Tight tercile: the top third of the real window's |r|, as kdiff.readouts.
- KD_ens: mean over pairs of real k04 minus the 'im' ensemble mean; KD_ens_ia the same against the 'ia' mean (kdiff's construction
  with its single draw replaced by the ensemble mean). KD0: real minus draw 0 of 'ia', which is kdiff's KD.
- Mechanism (M1, the spectral test, M2) in raw k04 / k22 units, never per-series SD units: per animal, the mean over drug units of
  (unit mean minus its session's eyes-closed mean); pooled = mean of the 4 animals. Spectral concentration holds if
  pooled change(pred) / pooled change(ia mean) of window-mean k04 is in [.75, 1.25] and its window r over all windows is >= .9;
  IAAFT-specific if the ratio is outside. The same against 'ft' is printed beside it.
- M2: the marginal term of single-draw KD is draw 0's net against 'ia' minus its net against 'im' = im_win[:, 0] - ia_win[:, 0]
  (window-mean k04); share = its pooled change over KD0's pooled change (per-animal shares are printed but unstable where KD0's change is near 0).
- M3: whether KZ (the null's own spectral fall inside the reference) still rises with drug, positive in 4/4 animals.
- Decision 5 compares KZ with KD and ZGK as kdiff scored them (recomputed here from kd_*.npz and zns_*.npz, gated against
  kd-score.json).
Gates: ids match across ke, kd and zns; draw 0's window-mean net k04 reproduces kd_*.npz KD within 1e-5; KD and ZGK through
level() reproduce kd-score.json.
  python kens_score.py <dir with ke_[AB]_*.npz> <dir with kd_[AB]_*.npz and kd-score.json> <zns dir>
"""
import sys, glob, json, numpy as np, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
ROWS = ('r', 'k40', 'k22', 'k04', 'M', 'tau_u', 'tau_v')                # kens.ROWS; pred is (k40, k22, k04)
I = {k: i for i, k in enumerate(ROWS)}
DRUG = ['KT', 'KTMD', 'MD', 'PF']; CW = ('Chibi', 'George')


def readouts(q):
    """Per-window readouts (dict of (n,) arrays) from one ke file; real, arm means/sds (n, 7, E), wins (n, B, 7), pred (n, 3, E)."""
    f = lambda k: q[k].astype(float); B = int(q['B']); real, absr = f('real'), np.abs(f('real')[:, 0])
    tight = absr >= np.quantile(absr, 2 / 3, axis=1)[:, None]; R, zero = {}, {}
    z = lambda a, row: (lambda sd: (np.where(sd > 0, (real[:, I[row]] - f(f'{a}_mean')[:, I[row]]) / np.where(sd > 0, sd, 1), np.nan),
                                    int((sd <= 0).sum())))(f(f'{a}_sd')[:, I[row]])
    for name, a, row in (('KZ', 'im', 'k04'), ('MZ', 'im', 'M'), ('K40Z', 'im', 'k40'), ('K22Z', 'im', 'k22'),
                         ('KZ_ia', 'ia', 'k04'), ('KZ_ft', 'ft', 'k04')):
        v, zero[name] = z(a, row); R[name] = np.nanmean(v, 1)
        if name == 'KZ': R['KZ_tight'] = np.array([w[s & np.isfinite(w)].mean() for w, s in zip(v, tight)])
    net = lambda a, row: (real[:, I[row]] - f(f'{a}_mean')[:, I[row]]).mean(1)
    R.update(KD_ens=net('im', 'k04'), KD_ens_ia=net('ia', 'k04'), MN=net('im', 'M'), K40N=net('im', 'k40'), K22N=net('im', 'k22'),
             TUN=net('im', 'tau_u'), TVN=net('im', 'tau_v'), TAIL=(q['im_below'][:, I['k04']] >= B - 2).mean(1),
             TAILM=(q['im_below'][:, I['M']] >= B - 2).mean(1))
    R['KD0'] = real[:, I['k04']].mean(1) - f('ia_win')[:, 0, I['k04']]
    raw = {f'{row}_{s}': v for row, j, pj in (('k04', I['k04'], 2), ('k22', I['k22'], 1)) for s, v in
           (('real', real[:, j].mean(1)), ('pred', f('pred')[:, pj].mean(1)),
            *((a, f(f'{a}_mean')[:, j].mean(1)) for a in ('ia', 'im', 'ft')))}
    raw.update(m2=f('im_win')[:, 0, I['k04']] - f('ia_win')[:, 0, I['k04']], kd0=R['KD0'],
               **{f'km_{s}': f('kmarg')[:, i].mean(1) for i, s in enumerate(('real', 'ia', 'im', 'ft', 'pred'))})
    kmgap = float(np.abs(f('kmarg')[:, 2] - f('kmarg')[:, 0]).max())
    return R, raw, zero, kmgap


def change(df, c):
    """Per-animal mean over drug units of (unit mean - its session's eyes-closed mean), in c's own units."""
    ref = df[df.state == 'awake_ec'].groupby('unit')[c].mean(); d = df.assign(d=df[c] - df.unit.map(ref))
    U = d.groupby(['unit', 'animal', 'group']).d.mean().reset_index(); return U[U.group.isin(DRUG)].groupby('animal').d.mean()


def level(df, c):
    """kdiff.score's panel (c) statistics for one column (copied, not imported, so kdiff.py stays as committed)."""
    from scipy.stats import ttest_1samp, mannwhitneyu
    ref = df[df.state == 'awake_ec'].groupby('unit')[c].mean(); dd = df[c] - df.unit.map(ref)
    s = dd / dd[df.state == 'awake_eo'].std(ddof=1); df[c + '_sd'] = s
    U = df.assign(s=s).groupby(['unit', 'animal', 'group']).s.mean().reset_index()
    dr = U[U.group.isin(DRUG)]; A = dr.groupby('animal').s.mean(); t = ttest_1samp(A, 0)
    sl = U[U.group == 'SLP'].groupby('animal').s.mean(); sign = np.sign(A.mean()); per = {}
    for an in CW:
        w = df[(df.animal == an)]; xs, xd = w[w.group == 'SLP'][c + '_sd'], w[w.state == 'drug'][c + '_sd']
        auc = mannwhitneyu(sign * xd, sign * xs).statistic / (len(xd) * len(xs))
        per[an] = dict(drug=float(A[an]), sleep=float(sl[an]), diff=float(A[an] - sl[an]), auc=float(auc),
                       agents={g: float(U[(U.animal == an) & (U.group == g)].s.mean()) for g in DRUG})
    w = df[df.animal.isin(CW) & df.group.isin(DRUG + ['SLP'])]
    X = np.column_stack([np.ones(len(w)), w.state == 'drug', w.animal == 'George', w.delta]); y = w[c + '_sd'].values
    b0 = np.linalg.lstsq(X[:, :3], y, rcond=None)[0][1]; b1 = np.linalg.lstsq(X, y, rcond=None)[0][1]
    rho = {g: float(w[w.group.isin(gs)][[c + '_sd', 'delta']].corr(method='spearman').iloc[0, 1])
           for g, gs in (('sleep', ['SLP']), ('drug', DRUG))}
    print(f'{c:10s} drug {A.mean():+6.2f} SD (animals {" ".join(f"{v:+.2f}" for v in A)}) t(3) {t.statistic:+6.2f} p {t.pvalue:.4f} | '
          f'sleep {" ".join(f"{v:+.2f}" for v in sl)} | drug-sleep ' + ' '.join(f'{an} {v["diff"]:+.2f} AUC {v["auc"]:.2f} '
          f'MD {v["agents"]["MD"]:+.2f}' for an, v in per.items()) + f' | β drug|δ {b0:+.2f}->{b1:+.2f} | ρδ sleep {rho["sleep"]:+.2f} drug {rho["drug"]:+.2f}')
    return dict(drug_animals=A.round(3).to_dict(), t3=float(t.statistic), p=float(t.pvalue), drug=float(A.mean()),
                sleep_animals=sl.round(3).to_dict(), groups=df.groupby('group')[c + '_sd'].mean().round(3).to_dict(),
                chibi_george=per, drug_vs_sleep_beta=float(b0), drug_vs_sleep_beta_given_delta=float(b1), delta_spearman=rho,
                level_ec=float(df[df.state == 'awake_ec'][c].mean()))


def score(d, kd, zd):
    import pandas as pd
    F, zeros, gates = [], {}, []
    for f in sorted(glob.glob(f'{d}/ke_[AB]_*.npz')):
        q, name = np.load(f), Path(f).name; k = np.load(Path(kd, name.replace('ke_', 'kd_'))); z = np.load(Path(zd, name.replace('ke_', 'zns_')))
        assert (q['id'] == k['id']).all() and (q['id'] == z['id']).all(), f
        R, raw, zero, kmgap = readouts(q); K = k['K'].astype(float); KD = (K[:, 0, 5] - K[:, 1, 5]).mean(1)
        gap = float(np.abs(R['KD0'] - KD).max()); gates.append(gap < 1e-5)
        print(f'gate {name}: max |draw-0 net k04 - kd KD| = {gap:.1e}', 'PASS' if gap < 1e-5 else 'FAIL',
              f'| max |kmarg im - real| {kmgap:.1e} | zero-sd pairs', {a: b for a, b in zero.items() if b})
        for a, b in zero.items(): zeros[a] = zeros.get(a, 0) + b
        F.append(pd.DataFrame(dict(id=q['id'], unit=q['unit'], state=q['state'], agent=q['agent'], animal=q['animal'],
                                   delta=z['delta_fraction'], KD=KD, ZGK=z['V'][:, 1, 0] - z['V'][:, 1, 1], **R,
                                   **{'raw_' + a: v for a, v in raw.items()})))
    df = pd.concat(F, ignore_index=True); assert df.id.is_unique
    df['group'] = np.where(df.state == 'drug', df.agent, np.where(df.state == 'sleep', 'SLP', df.state))
    B = int(np.load(f)['B']); print(f'B = {B}, windows {len(df)}, tail chance {3 / (B + 1):.3f}; zero-sd pairs in total', zeros)
    keys = ['KD', 'ZGK', 'KZ', 'MZ', 'KZ_tight', 'K40Z', 'K22Z', 'KZ_ia', 'KZ_ft', 'KD0', 'KD_ens', 'KD_ens_ia', 'MN',
            'K40N', 'K22N', 'TUN', 'TVN', 'TAIL', 'TAILM']
    out = {c: level(df, c) for c in keys}
    ref = json.loads(Path(kd, 'kd-score.json').read_text())
    pg = max(abs(out[c]['t3'] - ref[c]['t3']) + sum(abs(out[c]['chibi_george'][a]['auc'] - ref[c]['chibi_george'][a]['auc']) for a in CW)
             for c in ('KD', 'ZGK'))
    print(f'gate pipeline: KD and ZGK t(3) + AUCs against kd-score.json, max gap {pg:.1e}', 'PASS' if pg < 1e-9 else 'FAIL')
    print('\nmechanism, raw units: per-animal drug - eyes-closed change of window means (pooled = mean of animals)')
    ch = {c: change(df, 'raw_' + c) for c in [c[4:] for c in df.columns if c.startswith('raw_')]}
    for c, A in ch.items(): print(f'  {c:9s} pooled {A.mean():+.4f} | ' + ' '.join(f'{a} {v:+.4f}' for a, v in A.items()))
    mech = {}
    for row in ('k04', 'k22'):
        for a in ('ia', 'ft', 'im'):
            rat = ch[f'{row}_pred'].mean() / ch[f'{row}_{a}'].mean(); r = float(np.corrcoef(df[f'raw_{row}_pred'], df[f'raw_{row}_{a}'])[0, 1])
            mech[f'{row}_pred_vs_{a}'] = dict(ratio=float(rat), window_r=r, spectral=bool(.75 <= rat <= 1.25 and r >= .9))
            print(f'  M1 {row} pred / {a}: pooled ratio {rat:.3f}, window r {r:.3f}' + (f' -> spectral concentration {"HOLDS" if .75 <= rat <= 1.25 and r >= .9 else "FAILS"}'
                  f'{"" if .75 <= rat <= 1.25 else " (IAAFT-specific: outside ±25%)" if a == "ia" else ""}' if row == 'k04' and a != 'im' else ''))
    m2 = ch['m2'].mean() / ch['kd0'].mean(); mech['M2_marginal_share'] = float(m2)
    print(f'  M2 marginal term share of single-draw KD drug change: {m2:.3f} (animals ' +
          ' '.join(f'{a} {ch["m2"][a] / ch["kd0"][a]:+.2f}' for a in ch['m2'].index) + ')')
    kz = out['KZ']; pos = sum(v > 0 for v in kz['drug_animals'].values()); mech['M3_KZ_positive_animals'] = pos
    rep = kz['t3'] >= out['KD']['t3'] and pos == 4; beat = all(kz['chibi_george'][a]['auc'] > out['ZGK']['chibi_george'][a]['auc'] for a in CW)
    print(f'\ndecision: KZ t(3) {kz["t3"]:+.2f} vs KD {out["KD"]["t3"]:+.2f}, positive in {pos}/4 animals -> KZ '
          f'{"REPLACES" if rep else "does not replace"} KD as the level readout; AUC ' +
          ' '.join(f'{a} {kz["chibi_george"][a]["auc"]:.2f} vs Δz {out["ZGK"]["chibi_george"][a]["auc"]:.2f}' for a in CW) +
          f' -> {"improves on" if beat else "does not improve on"} Δz for anaesthesia against sleep' +
          ('' if pos == 4 else '; KZ not positive in 4/4: the level readouts\' drug effect is attributed to the null\'s spectral and marginal shifts'))
    C = df[[c + '_sd' for c in keys]].corr().loc['ZGK_sd']; print('window r with ZGK (sd units):', C.round(2).to_dict())
    Path(d, 'ke-score.json').write_text(json.dumps(dict(readouts=out, mechanism=mech, raw_changes={c: A.round(5).to_dict() for c, A in ch.items()},
                                                        decision=dict(replaces_KD=bool(rep), beats_dz=bool(beat)), gates_pass=bool(all(gates) and pg < 1e-9),
                                                        zero_sd=zeros, B=B), indent=1) + '\n')
    df.to_csv(Path(d, 'ke-windows.csv.gz'), index=False)


if __name__ == '__main__':
    score(*sys.argv[1:4])
