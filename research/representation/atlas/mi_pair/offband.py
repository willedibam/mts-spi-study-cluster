"""Off-band count of fig:mi panel (b) against chance, wake and sleep (rules: preregs/offband-prereg.md; result: offband-result.md).

    python offband.py <zns dir> [<offband dir with t1_*.npz, t2_*.npz>]  > offband-score.txt

Step 1 and the draw-0 tail check use only the zns arrays, so the script reruns unchanged on revised zns (e.g. a revised K).
The null steps need the K draws of offband_null.py, made with the same estimator: a revised K needs new draws.
"""
import sys, glob, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import ttest_1samp, spearmanr, norm, binom
import statsmodels.formula.api as smf

Q, NP = 95, 496
GROUPS = ('awake_eo', 'awake_ec', 'recovery_ec', 'SLP', 'KT', 'KTMD', 'MD', 'PF')
DRUGS = ('KT', 'KTMD', 'MD', 'PF')
PANEL_B = {'KTMD': '20110114ktmdanesthesiaandsleepgeorgetoruyanagawa:drug:08',
           'awake_eo': '20120717slpanesthesiaandsleepchibitoruyanagawa:awake_eo:08',
           'SLP': '20120706slpanesthesiaandsleepchibitoruyanagawa:sleep:15',
           'KT': '20120719ktanesthesiaandsleepchibitoruyanagawa:drug:03',
           'PF': '20120730pfanesthesiaandsleepchibitoruyanagawa:drug:08'}


def resid(g, k):
    return k - np.polyval(np.polyfit(g, k, 1), g)


def load(zdir):
    rows, E = [], []
    for p in sorted(Path(zdir).glob('zns_[AB]_*.npz')):
        with np.load(p) as d:
            f = pd.DataFrame({c: d[c] for c in ('id', 'unit', 'animal', 'state', 'agent')}); f['file'] = p.stem; f['k'] = np.arange(len(f))
            f['zr'], f['zs'] = d['V'][:, 1, 0], d['V'][:, 1, 1]; rows.append(f); E.append(d['E'][:, 1].astype(float))
    df = pd.concat(rows, ignore_index=True); E = np.concatenate(E)            # E: (n, rec/sur, G/K, 496)
    assert df.id.is_unique and np.array_equal(E[:, 0, 0], E[:, 1, 0])       # G_surr = G_rec (spectrum-final IAAFT keeps r)
    df['group'] = np.where(df.state == 'drug', df.agent, np.where(df.state == 'sleep', 'SLP', df.state))
    df['res'] = df.zr - df.zs
    er = np.array([resid(e[0, 0], e[0, 1]) for e in E]); e0 = np.array([resid(e[1, 0], e[1, 1]) for e in E])
    df['L'] = np.percentile(np.abs(e0), Q, axis=1); df['c'] = (np.abs(er) > df.L.values[:, None]).sum(1)
    return df, E, er, e0


def session(df, col):
    """Per session (unit): state means minus awake_ec mean, as montage/animal_level.py."""
    out = []
    for u, d in df.groupby('unit'):
        b = d[d.state == 'awake_ec'][col].mean(); r = dict(unit=u, agent=d.agent.iloc[0], animal=d.animal.iloc[0])
        for s in ('drug', 'sleep', 'awake_eo', 'recovery_ec'):
            if (d.state == s).any(): r[s] = d[d.state == s][col].mean() - b
        out.append(r)
    s = pd.DataFrame(out); s['target'] = s.drug.combine_first(s.sleep); return s


def animal_test(S, label):
    dr = S[S.agent != 'SLP']; A = dr.groupby('animal').target.mean(); E = dr.groupby('animal').awake_eo.mean()
    t = ttest_1samp(A.values, 0, alternative='greater'); n = len(A)
    m = smf.mixedlm('target ~ 1', dr, groups='animal').fit(reml=True)
    slp = S[S.agent == 'SLP'].target
    print(f'{label}: animal means (drugs pooled) {A.round(2).to_dict()}; {int((A > 0).sum())}/{n} positive, sign p '
          f'{binom.sf((A > 0).sum() - 1, n, .5):.4f}; one-sided t({n - 1}) = {t.statistic:.2f}, p = {t.pvalue:.4f}')
    print(f'  mixed model ({len(dr)} drug sessions, random animal intercept) {m.params.Intercept:+.2f} ± {m.bse.Intercept:.2f};'
          f' sessions up {int((dr.target > 0).sum())}/{len(dr)}; eyes-open control per animal {E.round(2).to_dict()}')
    print(f'  per agent (session mean, sessions up): ' + ', '.join(
        f'{a} {S[S.agent == a].target.mean():+.2f} ({int((S[S.agent == a].target > 0).sum())}/{(S.agent == a).sum()})' for a in DRUGS))
    print(f'  sleep {slp.mean():+.2f} ({int((slp > 0).sum())}/{len(slp)} up; animals {S[S.agent == "SLP"].groupby("animal").target.mean().round(2).to_dict()}),'
          f' ratio to pooled drug session mean {slp.mean() / dr.target.mean():+.2f}; recovery (drug sessions) {dr.recovery_ec.mean():+.2f}')
    return dict(animal=A.to_dict(), t=t.statistic, p=t.pvalue, mixed=(m.params.Intercept, m.bse.Intercept), sleep=slp.mean(),
                drug_sessions=dr.target.mean(), eo=E.to_dict())


def step1(df):
    print('== Step 1: off-band count per window (band() of plot_character_draft.py; 496 pairs, nominal 24.8) ==')
    eo = df[df.group == 'awake_eo'].groupby('animal').c.mean()
    print(f'{"group":12s} {"n":>4s} {"mean":>6s} {"median":>6s} {"over 24.8":>9s} {"over own-animal wake":>20s}  per animal (mean c)')
    for g in GROUPS:
        d = df[df.group == g]; ow = (d.c - d.animal.map(eo)).mean()
        print(f'{g:12s} {len(d):4d} {d.c.mean():6.1f} {d.c.median():6.0f} {d.c.mean() - 24.8:+9.1f} {ow:+20.1f}  {d.groupby("animal").c.mean().round(1).to_dict()}')
    print('per session (unit: agent animal, mean c by state):')
    for u, d in df.groupby('unit'):
        print(f'  {d.agent.iloc[0]:5s} {d.animal.iloc[0]:6s} {u[:8]} ' + ' '.join(f'{s} {d[d.state == s].c.mean():5.1f}' for s in
                                                                                ('awake_ec', 'awake_eo', 'drug', 'sleep', 'recovery_ec') if (d.state == s).any()))
    S = session(df, 'c'); r = animal_test(S, 'Δc (session state − awake_ec mean)')
    Sz = session(df, 'res'); rho_w = spearmanr(df.c, df.res).statistic; rho_s = spearmanr(S.target, Sz.target).statistic
    rho_in = np.mean([spearmanr(d.c, d.res).statistic for _, d in df.groupby('group')])
    print(f'count against Δz: window Spearman(c, z_rec − z_surr) {rho_w:+.3f} (mean within state group {rho_in:+.3f}); '
          f'session Spearman(Δc, Δz) {rho_s:+.3f}')
    print('panel (b) windows: ' + ', '.join(f'{g} c={int(df[df.id == w].c.iloc[0])}' for g, w in PANEL_B.items()))
    return S, r


def tails(df, er, e0, label='draw 0'):
    """Step 4 tail and heteroscedasticity check of the surrogate residuals (standardised per window by SD)."""
    print(f'== Step 4: surrogate residual tails ({label}; per-window SD-standardised, two-sided) ==')
    u = e0 / e0.std(1, keepdims=True); qs = (.05, .01, .001, .05 / NP)
    print(f'{"group":12s} ' + ' '.join(f'P(|u|>{norm.isf(q / 2):.2f}) [{q}]' for q in qs) + '   excess kurt  rho(|e|,G)  SD top/bottom G tercile')
    for g in GROUPS + ('all',):
        m = (df.group == g).values if g != 'all' else np.ones(len(df), bool); x = u[m].ravel()
        het = [spearmanr(np.abs(e0[i]), G[i]).statistic for i in np.where(m)[0]]
        rat = [np.std(e0[i][G[i] >= np.quantile(G[i], 2 / 3)]) / np.std(e0[i][G[i] <= np.quantile(G[i], 1 / 3)]) for i in np.where(m)[0]]
        print(f'{g:12s} ' + ' '.join(f'{np.mean(np.abs(x) > norm.isf(q / 2)):19.4f}' for q in qs) +
              f'   {((x ** 4).mean() - 3):9.2f}  {np.median(het):+9.2f}  {np.median(rat):9.2f}')
    return u


def scale_shape(df, er, e0):
    """Post hoc, descriptive: is a window's count its recording's residual scale (SD over the surrogate's) or its tail shape?
    Gaussian prediction from scale alone: 496 * 2 * Phi(-L / SD_rec)."""
    print('== Post hoc: count from residual scale against tail shape ==')
    s = er.std(1) / e0.std(1); pred = NP * 2 * norm.sf(df.L.values / er.std(1))
    kr = (er / er.std(1, keepdims=True)) ** 4; kr = kr.mean(1) - 3
    print(f'{"group":12s} {"SD rec/sur":>10s} {"c":>6s} {"c from scale":>12s} {"rec exc. kurt":>13s}')
    for g in GROUPS:
        m = (df.group == g).values
        print(f'{g:12s} {np.median(s[m]):10.3f} {df.c[m].mean():6.1f} {pred[m].mean():12.1f} {np.median(kr[m]):13.2f}')
    print(f'window Spearman(c, SD ratio) {spearmanr(df.c, s).statistic:+.3f}')


def holm(p, a=.05):
    o = np.argsort(p); ok = p[o] <= a / (len(p) - np.arange(len(p))); return int(np.argmin(ok)) if not ok.all() else len(p)


def bh(p, q=.05, by=False):
    m = len(p); c = np.sum(1 / np.arange(1, m + 1)) if by else 1; ps = np.sort(p); ok = np.where(ps <= q * np.arange(1, m + 1) / (m * c))[0]
    return int(ok[-1] + 1) if len(ok) else 0


def per_pair(p, a=.05):
    """Discoveries by Bonferroni, Holm, BH, BY and Storey's non-null share 1 - pi0 (lambda .5)."""
    return dict(bonf=int(np.sum(p <= a / len(p))), holm=holm(p, a), bh=bh(p, a), by=bh(p, a, True),
                storey=1 - min(1, np.mean(p > .5) / .5))


def load_null(df, E, odir, tier):
    fs = sorted(Path(odir).glob(f'{tier}_*.npz')); assert fs, f'no {tier} files in {odir}'
    L = [np.load(f) for f in fs]; ids = np.concatenate([d['id'] for d in L]); pos = df.reset_index().set_index('id').loc[ids, 'index'].values
    assert np.array_equal(df.file.values[pos], np.concatenate([d['zfile'] for d in L])) and np.array_equal(df.k.values[pos], np.concatenate([d['k'] for d in L]))
    K = np.concatenate([d['K'] for d in L]).astype(float)
    dev = np.abs(K[:, 0] - E[pos, 1, 1]).max(); print(f'{tier} reproduction gate: max |K draw 0 − zns K_surr| = {dev:.2e} over {len(pos)} windows',
                                                       'PASS' if dev < 1e-4 else 'FAIL')
    assert dev < 1e-4, 'gate failed: stop (prereg)'
    Gc = E[pos, 0, 0] - E[pos, 0, 0].mean(1, keepdims=True); K -= K.mean(2, keepdims=True)   # residual about each draw's own line
    K -= ((K * Gc[:, None]).sum(2) / (Gc ** 2).sum(1)[:, None])[..., None] * Gc[:, None]; e = K
    return pos, e, np.concatenate([d['W_rec'] for d in L]).astype(int), np.concatenate([d['W_s0'] for d in L]).astype(int)


def step2(df, E, er, odir):
    print('== Step 2: the count under surrogate against surrogate (99 draws per window, about their own lines, draw 0\'s limit) ==')
    pos, e, Wr, Ws = load_null(df, E, odir, 't1'); L = df.L.values[pos]
    assert np.allclose(np.percentile(np.abs(e[:, 0]), Q, axis=1), L)
    c0 = (np.abs(e[:, 1:]) > L[:, None, None]).sum(2); c = df.c.values[pos]; B = c0.shape[1]
    d = df.iloc[pos].copy(); d['m0'], d['s0'] = c0.mean(1), c0.std(1, ddof=1); d['E'] = c - d.m0
    d['p'] = (1 + (c0 >= c[:, None]).sum(1)) / (B + 1)
    print(f'pooled null count: mean {c0.mean():.2f}, SD within window {np.sqrt((c0.var(1, ddof=1)).mean()):.2f} (Binomial 4.85; '
          f'inflation {np.sqrt((c0.var(1, ddof=1)).mean()) / 4.85:.2f}), 2.5–97.5% {np.percentile(c0, [2.5, 97.5])}')
    print(f'{"group":12s} {"null mean":>9s} {"null SD":>7s} {"c":>6s} {"E":>6s} {"share p<.05":>11s}  verdict')
    V = {}
    for g in GROUPS:
        x = d[d.group == g]; sh = np.mean(x.p < .05); V[g] = x.E.mean()
        print(f'{g:12s} {x.m0.mean():9.2f} {x.s0.mean():7.2f} {x.c.mean():6.1f} {x.E.mean():+6.1f} {sh:11.3f}  {"at chance" if sh <= .10 else "in excess"}')
    S = session(d, 'E'); r = animal_test(S, 'Δ excess E (session state − awake_ec mean)')
    dr = d[d.group.isin(DRUGS)].E.mean()
    ok = r['p'] < .05 and dr > 0 and dr >= 3 * V['awake_eo'] and dr >= 3 * V['SLP']
    print(f'decision: (i) animal test p = {r["p"]:.4f}; (ii) pooled drug E {dr:+.1f} against 3 x wake {3 * V["awake_eo"]:+.1f} and 3 x sleep '
          f'{3 * V["SLP"]:+.1f} -> claim {"HOLDS" if ok else "DOES NOT HOLD"}')
    print('decomposition by tercile (off-band share, recording against null draws), within each window:')
    for name, X in (('G', E[pos, 0, 0]), ('Theiler W_rec', Wr), ('W_rec − W_surr0', Wr - Ws)):
        for g in ('awake_eo', 'SLP', 'KTMD', 'PF'):
            m = (d.group == g).values; out = []
            for t in range(3):
                rr, nn = [], []
                for w in np.where(m)[0]:
                    q = np.quantile(X[w], [t / 3, (t + 1) / 3]); sel = (X[w] >= q[0]) & (X[w] <= q[1])
                    rr.append(np.mean(np.abs(er[pos[w]][sel]) > L[w])); nn.append(np.mean(np.abs(e[w, 1:, sel]) > L[w]))
                out.append(f'{np.mean(rr):.3f}/{np.mean(nn):.3f}')
            print(f'  {name:16s} {g:9s} low→high: ' + '  '.join(out))
    print(f'Theiler window: recording mean {Wr.mean():.1f}, draw 0 {Ws.mean():.1f}; share of pairs differing {np.mean(Wr != Ws):.2f}')
    return d, pos, e


def step3(d, df, er, e, pos):
    print('== Step 3: per-pair procedures, all windows, null pooled over pairs and the 100 draws (min p 2.0e-5) ==')
    rows = []
    for w in range(len(pos)):
        pool = np.sort(np.abs(e[w]).ravel()); a = np.abs(er[pos[w]]); n = len(pool)
        p = (1 + n - np.searchsorted(pool, a, side='left')) / (n + 1)
        r = per_pair(p); mx = np.abs(e[w]).max(1); r['wy'] = int(np.sum(a > np.percentile(mx, 95)))
        sd0 = e[w, 0].std(); r['gauss_bonf'] = int(np.sum(a > norm.isf(.025 / NP) * sd0))
        r['gauss_fwer'] = np.mean((np.abs(e[w, 1:]) > norm.isf(.025 / NP) * sd0).any(1))
        u = a / sd0; r['own_bulk'] = int(np.sum(u > 1.96 * 1.4826 * np.median(u)))    # Efron-style: the window's own central scale
        rows.append(r)
    R = pd.DataFrame(rows, index=d.index); d = d.join(R)
    uw = np.concatenate([np.abs(er[pos[w]]) / e[w, 0].std() for w in np.where(d.group == 'awake_eo')[0]]); s0 = 1.4826 * np.median(uw)
    d['wake_null'] = [int(np.sum(np.abs(er[pos[w]]) / e[w, 0].std() > 1.96 * s0)) for w in range(len(pos))]
    print(f'empirical null from awake_eo recording residuals: sigma0 = {s0:.3f} surrogate SD')
    cols = ['c', 'bonf', 'holm', 'wy', 'bh', 'by', 'storey', 'wake_null', 'own_bulk', 'gauss_bonf', 'gauss_fwer']
    print(d.groupby('group')[cols].mean().loc[list(GROUPS)].round(3).to_string())
    return d


def tier2(df, E, er, odir):
    print('== Step 3, panel (b) windows: each pair its own 20,000-draw null ==')
    pos, e, Wr, Ws = load_null(df, E, odir, 't2')
    for w in range(len(pos)):
        a = np.abs(er[pos[w]]); B = e.shape[1]; sd = e[w].std(0)
        p = (1 + (np.abs(e[w]) >= a).sum(0)) / (B + 1); r = per_pair(p)
        t = np.abs(e[w]) / sd; r['wy'] = int(np.sum(a / sd > np.percentile(t.max(1), 95)))
        r['gauss_fwer'] = np.mean((np.abs(e[w]) > norm.isf(.025 / NP) * e[w, 0].std()).any(1))
        pool = np.sort(np.abs(e[w]).ravel()); pp = (1 + pool.size - np.searchsorted(pool, a)) / (pool.size + 1)
        cs = np.corrcoef(e[w].T); iu = np.triu_indices(32, 1); pr = list(zip(*iu)); sh = np.array([[len({*pr[i]} & {*pr[j]}) > 0 for j in range(NP)] for i in range(NP)])
        off = ~np.eye(NP, dtype=bool)
        print(f'{df.group.iloc[pos[w]]:9s} c={df.c.iloc[pos[w]]:3d} ' + ' '.join(f'{k} {v:.3g}' for k, v in r.items()) +
              f'; per-pair null SD CV {sd.std() / sd.mean():.3f}; Spearman(per-pair p, pooled p) {spearmanr(p, pp).statistic:.3f};'
              f' null corr shared-channel {cs[sh & off].mean():+.3f}, disjoint {cs[~sh & off].mean():+.3f}')


def channels(df, er, e0):
    """Step 5 check: share of each window's log squared standardised residual (and off-band indicator) that is channel-additive
    (u_i + u_j, 32 parameters; chance R2 31/495 = .063). Near chance means crossed channel random effects would pool nothing."""
    print('== Step 5 check: channel-additive structure of the residual magnitude (median R2 over windows; chance .063) ==')
    iu = np.triu_indices(32, 1); X = np.zeros((NP, 32)); X[np.arange(NP), iu[0]] = 1; X[np.arange(NP), iu[1]] = 1; P = X @ np.linalg.pinv(X)
    r2 = lambda y: 1 - (((y - y.mean()) - P @ (y - y.mean())) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    for g in GROUPS:
        m = np.where(df.group == g)[0]; sd = lambda w: e0[w].std()
        print(f'{g:12s} log u^2: recording {np.median([r2(np.log((er[w] / sd(w)) ** 2)) for w in m]):.3f}, surrogate '
              f'{np.median([r2(np.log((e0[w] / sd(w)) ** 2)) for w in m]):.3f}; off-band indicator '
              f'{np.nanmedian([r2((np.abs(er[w]) > df.L.values[w]).astype(float)) for w in m]):.3f}')
    lu = np.log((er / e0.std(1, keepdims=True)) ** 2); print(f'between-window share of var(log u^2): {lu.mean(1).var() / lu.var():.4f}')


if __name__ == '__main__':
    df, E, er, e0 = load(sys.argv[1]); G = E[:, 0, 0]
    S, r1 = step1(df)
    tails(df, er, e0)
    scale_shape(df, er, e0)
    channels(df, er, e0)
    if len(sys.argv) > 2:
        d, pos, e = step2(df, E, er, sys.argv[2]); step3(d, df, er, e, pos)
        if list(Path(sys.argv[2]).glob('t2_*.npz')): tier2(df, E, er, sys.argv[2])
