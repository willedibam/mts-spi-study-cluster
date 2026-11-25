"""Mean difference kurtosis: per window, the mean over the 496 channel pairs of the excess (Fisher) kurtosis of the electrode
difference a - b, the statistic panel (a)'s strip prints for one pair, as a level readout beside the anchor's Δz (post hoc,
25 November; rule ../preregs/kdiff-prereg.md).

Same rows, 32-channel subsets, normal-scoring and IAAFT draw as run.py and split.py. Per pair, the standardised principal axes
u = (a + b) / sqrt(2(1 + r)), v = (a - b) / sqrt(2(1 - r)) give the five fourth cumulants k40, k31, k22, k13, k04. k04 is
scipy.stats.kurtosis(a - b) exactly (scale-free; v has mean 0 and variance 1 by construction). With normal-scored marginals
(marginal kappa4 = 0) it is a pure cross-cumulant of the pair: (6 k22' - 4 k31' - 4 k13') / (2 - 2r)^2 in the raw axes.
Saved per edge for the real window and its surrogate, with r, and k04 of the window before normal-scoring (real only).
  python kdiff.py <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers>
  python kdiff.py score <dir with kd_[AB]_*.npz> <zns dir> [<dir with split_[AB]_*.npz>, for the reproduction gate]
"""
import sys, glob, json, numpy as np, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
warnings.filterwarnings('ignore')
GLYPH = ('20110621ktmdanesthesiaandsleepchibitoruyanagawa:drug:02', 407)      # plot_character_draft.GLYPH_PAIR


def k4(x):
    """(6, E): r and k40, k31, k22, k13, k04 of every pair (i < j) in its principal axes."""
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu, ju = np.triu_indices(M, 1)
    r = (x.T @ x / T)[iu, ju]; a, b = x[:, iu], x[:, ju]
    u, v = (a + b) / np.sqrt(2 * (1 + r)), (a - b) / np.sqrt(2 * (1 - r))
    m = lambda p, q: np.mean(u ** p * v ** q, 0); m20, m02, m11 = m(2, 0), m(0, 2), m(1, 1)
    return np.array([r, m(4, 0) - 3 * m20 ** 2, m(3, 1) - 3 * m20 * m11, m(2, 2) - m20 * m02 - 2 * m11 ** 2,
                     m(1, 3) - 3 * m02 * m11, m(0, 4) - 3 * m02 ** 2])


def one(args):
    from run import ns
    from mtsspi.surrogates import multivariate_iaaft
    x, seed = args; v = ns(x); s = multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0]
    return np.array([k4(v), k4(s)], np.float32), k4(x)[5].astype(np.float32)


def readouts(K, raw):
    """Per-window readouts from (n, 2 real/surr, 6, E) cumulants; net = real - surrogate unless named otherwise."""
    r, k04 = np.abs(K[:, 0, 0]), K[:, :, 5]; net = k04[:, 0] - k04[:, 1]
    mardia = K[:, :, 1] + 2 * K[:, :, 3] + K[:, :, 5]                     # bivariate (Mardia) excess kurtosis, rotation-free
    q = np.quantile(r, [1 / 3, 2 / 3], axis=1).T; lo, hi = r < q[:, :1], r >= q[:, 1:]
    strat = lambda m: np.array([w[s].mean() for w, s in zip(net, m)])
    return dict(KD=net.mean(1), KD_real=k04[:, 0].mean(1), KD_surr=k04[:, 1].mean(1),
                KDmed=np.median(k04[:, 0], 1) - np.median(k04[:, 1], 1), KD_loose=strat(lo), KD_mid=strat(~lo & ~hi),
                KD_tight=strat(hi), MARDIA=(mardia[:, 0] - mardia[:, 1]).mean(1), KD_raw=raw.mean(1))


def score(d, zd, sd=None):
    import pandas as pd
    from scipy.stats import ttest_1samp, mannwhitneyu
    F, Z = [], []
    for f in sorted(glob.glob(f'{d}/kd_[AB]_*.npz')):
        q = np.load(f); z = np.load(f.replace(d, zd).replace('/kd_', '/zns_')); assert (q['id'] == z['id']).all(), f
        K = q['K'].astype(float); R = readouts(K, q['raw'].astype(float))
        R['ZGK'] = z['V'][:, 1, 0] - z['V'][:, 1, 1]
        F.append(pd.DataFrame(dict(id=q['id'], unit=q['unit'], state=q['state'], agent=q['agent'], animal=q['animal'],
                                   delta=z['delta_fraction'], **R)))
        if sd and Path(g := f.replace(d, sd).replace('/kd_', '/split_')).exists():   # gate: Kdiff_all = mean k04^2 / 48, net
            kd = (K[:, :, 5] ** 2 / 48).mean(2); gap = np.max(np.abs(kd[:, 0] - kd[:, 1] - np.load(g)['C'][:, 1]))
            print(f'gate {Path(f).name}: max |Kdiff recomputed - split.py| = {gap:.1e}', 'PASS' if gap < 1e-5 else 'FAIL')
        if GLYPH[0] in q['id']:
            w = list(q['id']).index(GLYPH[0]); print(f'glyph pair {GLYPH}: k04 real {K[w, 0, 5, GLYPH[1]]:.3f} surrogate {K[w, 1, 5, GLYPH[1]]:.3f}')
    df = pd.concat(F, ignore_index=True); assert df.id.is_unique
    df['group'] = np.where(df.state == 'drug', df.agent, np.where(df.state == 'sleep', 'SLP', df.state))
    keys = [c for c in df.columns if c[:2] in ('KD', 'MA', 'ZG')]; out, cw = {}, ('Chibi', 'George')
    for c in keys:                                                            # panel (c)'s units: window minus its session's
        ref = df[df.state == 'awake_ec'].groupby('unit')[c].mean(); dd = df[c] - df.unit.map(ref)   # eyes-closed mean, in
        s = dd / dd[df.state == 'awake_eo'].std(ddof=1); df[c + '_sd'] = s                       # eyes-open window SDs
        U = df.assign(s=s).groupby(['unit', 'animal', 'group']).s.mean().reset_index()
        dr = U[U.group.isin(['KT', 'KTMD', 'MD', 'PF'])]; A = dr.groupby('animal').s.mean(); t = ttest_1samp(A, 0)
        sl = U[U.group == 'SLP'].groupby('animal').s.mean(); sign = np.sign(A.mean())
        per = {}
        for an in cw:                                                         # drug against sleep, within the two animals
            w = df[(df.animal == an)]; xs, xd = w[w.group == 'SLP'][c + '_sd'], w[w.state == 'drug'][c + '_sd']
            auc = mannwhitneyu(sign * xd, sign * xs).statistic / (len(xd) * len(xs))
            per[an] = dict(drug=float(A[an]), sleep=float(sl[an]), diff=float(A[an] - sl[an]), auc=float(auc),
                           agents={g: float(U[(U.animal == an) & (U.group == g)].s.mean()) for g in ('KT', 'KTMD', 'MD', 'PF')})
        w = df[df.animal.isin(cw) & df.group.isin(['KT', 'KTMD', 'MD', 'PF', 'SLP'])]
        X = np.column_stack([np.ones(len(w)), w.state == 'drug', w.animal == 'George', w.delta]); y = w[c + '_sd'].values
        b0 = np.linalg.lstsq(X[:, :3], y, rcond=None)[0][1]; b1 = np.linalg.lstsq(X, y, rcond=None)[0][1]
        rho = {g: float(w[w.group.isin(gs)][[c + '_sd', 'delta']].corr(method='spearman').iloc[0, 1])
               for g, gs in (('sleep', ['SLP']), ('drug', ['KT', 'KTMD', 'MD', 'PF']))}
        out[c] = dict(drug_animals=A.round(3).to_dict(), t3=float(t.statistic), p=float(t.pvalue), drug=float(A.mean()),
                      sleep_animals=sl.round(3).to_dict(), groups=df.groupby('group')[c + '_sd'].mean().round(3).to_dict(),
                      chibi_george=per, drug_vs_sleep_beta=float(b0), drug_vs_sleep_beta_given_delta=float(b1), delta_spearman=rho,
                      level_ec=float(df[df.state == 'awake_ec'][c].mean()))
        print(f'{c:9s} drug {A.mean():+6.2f} SD (animals {" ".join(f"{v:+.2f}" for v in A)}) t(3) {t.statistic:+6.2f} p {t.pvalue:.4f} | '
              f'sleep {" ".join(f"{v:+.2f}" for v in sl)} | drug-sleep ' + ' '.join(f'{an} {v["diff"]:+.2f} AUC {v["auc"]:.2f} '
              f'MD {v["agents"]["MD"]:+.2f}' for an, v in per.items()) + f' | β drug|δ {b0:+.2f}->{b1:+.2f} | ρδ sleep {rho["sleep"]:+.2f} drug {rho["drug"]:+.2f}')
    C = df[[c + '_sd' for c in keys]].corr().loc['ZGK_sd']; print('window r with ZGK (all windows, sd units):', C.round(2).to_dict())
    Path(d, 'kd-score.json').write_text(json.dumps(out, indent=1) + '\n'); df.to_csv(Path(d, 'kd-windows.csv.gz'), index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'score': score(*sys.argv[2:5]); sys.exit()
    from multiprocessing import Pool
    from edges import sub32, meta, STATES
    d, out, agent, rowmode, nw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    man = json.load(open(f'{d}/manifest.json')); Wn = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    with Pool(nw) as p: res = p.map(one, [(Wn[r['id']].astype(float)[:, sub(r)], k) for k, r in enumerate(rows)], chunksize=1)
    m, _ = meta(rows, man); np.savez(out, K=np.array([a for a, _ in res]), raw=np.array([b for _, b in res]), **m); print('done', out, flush=True)
