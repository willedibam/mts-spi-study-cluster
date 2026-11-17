"""Frozen rule of amplitude-prereg.md. python gate_sum.py gate_mouse.npz gate_macaque.npz"""
import sys, json, numpy as np
from scipy.stats import binom, spearmanr
NAMES = ('gate', 'imbal', 'switch'); SECOND = 'sub-569073_ses-20210416'; rng = np.random.default_rng(0)
boot = lambda d: np.percentile(rng.choice(d, (10000, len(d))).mean(1), [2.5, 97.5])
kneed = lambda n: min(j for j in range(n + 1) if binom.sf(j - 1, n, .5) <= .025)


def verdict(d, sd, tag):
    n = len(d); k = kneed(n); neg, pos = int((d < 0).sum()), int((d > 0).sum()); lo, hi = boot(d); m = d.mean()
    if neg >= k and m < -2 * sd and hi < 0: v = 'DECOUPLING'
    elif pos >= k and m > 2 * sd and lo > 0: v = 'GAIN'
    elif abs(m) < 2 * sd and lo > -.05 and hi < .05: v = 'NEITHER'
    else: v = 'INCONCLUSIVE'
    print(f'{tag}: n={n}, negative {neg}, positive {pos} (need {k}), mean {m:+.4f} [{lo:+.4f}, {hi:+.4f}] = {m / sd:+.1f} null SD → {v}')


# ---- mouse (confirmatory) ----
G = np.load(sys.argv[1]); R = json.loads(str(G['rows'])); F = G['F']; res = F[:, 0] - F[:, 1:].mean(1)   # windows × 3 readouts
Zi = np.load('claar_z_ind.npz'); zr = Zi['ns'][:, 0] - Zi['ns'][:, 1:].mean(1); Rz = json.loads(str(Zi['rows']))
zmap, cnt = {}, {}
for j, r in enumerate(Rz):
    i = cnt.get(r['session'], 0); cnt[r['session']] = i + 1; zmap[(r['session'], i)] = zr[j]
g = lambda k: np.array([r.get(k, np.nan) for r in R])
ses, src, st, bi, df, t0, idx = g('session'), g('src'), g('state'), g('burst').astype(float), g('delta_fraction').astype(float), g('start_s').astype(float), g('idx')
D, SH, IS, LINK, SEC = [], [], [], [], {n: [] for n in NAMES}
for s in dict.fromkeys(ses[src == 'claar_ind_sel']):
    if s == SECOND: continue
    a = (ses == s) & (src == 'claar_sel') & (st == 'awake'); i = (ses == s) & (src == 'claar_ind_sel')
    tg = i & (bi >= np.percentile(bi[a], 5)) & (df >= np.percentile(df[a], 95))
    if tg.sum() < 3: continue
    D.append(res[tg, 0].mean() - res[a, 0].mean())
    for q, n in enumerate(NAMES): SEC[n].append(res[tg, q].mean() - res[a, q].mean())
    ai = np.flatnonzero(a)[np.argsort(t0[a])]; h = len(ai) // 2; SH.append(res[ai[h:], 0].mean() - res[ai[:h], 0].mean())
    ti = np.flatnonzero(tg)
    if len(ti) >= 5: LINK.append(spearmanr(res[ti, 0], [zmap[(s, int(idx[w]))] for w in ti])[0])
    print(f'  {s}: target {tg.sum():2d}, Δgate {D[-1]:+.4f}, split-half {SH[-1]:+.4f}')
D = np.array(D); sd = np.std(SH, ddof=1); print(f'mouse awake split-half null SD {sd:.4f} (n={len(SH)})')
verdict(D, sd, 'PRIMARY mouse induction Δgate')
for n in NAMES[1:]: print(f'  secondary Δ{n}: mean {np.mean(SEC[n]):+.4f}, negative {sum(v < 0 for v in SEC[n])}/{len(SEC[n])}')
for s in dict.fromkeys(ses[src == 'claar_sel']):
    if s == SECOND: continue
    a = (ses == s) & (src == 'claar_sel') & (st == 'awake'); b = (ses == s) & (src == 'claar_sel') & (st == 'iso')
    if b.sum() >= 3 and a.sum() >= 3: IS.append(res[b, 0].mean() - res[a, 0].mean())
IS = np.array(IS); lo, hi = boot(IS)
print(f'SIGN CONTROL isoflurane epoch − awake: n={len(IS)}, positive {(IS > 0).sum()}, mean {IS.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] → {"behaves (gain +)" if IS.mean() > 0 and lo > 0 else "NOT as predicted: primary not interpreted"}')
print(f'secondary link, within-mouse Spearman(gate res, z res) over target windows: median {np.median(LINK):+.2f}, positive {sum(v > 0 for v in LINK)}/{len(LINK)}')

# ---- macaque (disclosed extension) ----
if len(sys.argv) > 2:
    G = np.load(sys.argv[2]); R = json.loads(str(G['rows'])); F = G['F']; res = F[:, 0] - F[:, 1]
    zres = {}
    for f in dict.fromkeys(r['zfile'] for r in R):
        V = np.load(f)['V']; zres[f] = V[:, 1, 0] - V[:, 1, 1]
    zr = np.array([zres[r['zfile']][r['k']] for r in R]); g = lambda k: np.array([r[k] for r in R])
    un, st, ag, an = g('unit'), g('state'), g('agent'), g('animal'); U = {}
    for u in dict.fromkeys(un):
        m = lambda s: (un == u) & (st == s); b = res[m('awake_ec'), 0].mean()
        U[u] = dict(ag=ag[un == u][0], an=an[un == u][0], **{s: res[m(s), 0].mean() - b for s in ('drug', 'sleep', 'awake_eo', 'recovery_ec') if m(s).any()})
        dw = np.flatnonzero(m('drug'))
        if len(dw) >= 5: U[u]['link'] = spearmanr(res[dw, 0], zr[dw])[0]
    sd = np.std([U[u]['awake_eo'] for u in U if 'awake_eo' in U[u]], ddof=1); print(f'\nmacaque eyes-open null SD {sd:.4f}')
    for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP'):
        s = 'sleep' if a == 'SLP' else 'drug'; d = np.array([U[u][s] for u in U if U[u]['ag'] == a and s in U[u]])
        print(f'  {a:5s} n={len(d):2d} Δgate {d.mean():+.4f} ({d.mean() / sd:+.1f} SD), negative {(d < 0).sum()}/{len(d)}')
    d = np.array([U[u]['drug'] for u in U if 'drug' in U[u]]); verdict(d, sd, 'macaque pooled drug Δgate')
    gp = [U[u]['drug'] for u in U if U[u]['an'] == 'George' and U[u]['ag'] == 'PF' and 'drug' in U[u]]; print(f'  George PF (seen before): {np.round(gp, 4)}')
    L = [U[u]['link'] for u in U if 'link' in U[u] and U[u]['ag'] != 'SLP']; print(f'  link within drug sessions, Spearman(gate res, z res): median {np.median(L):+.2f}, positive {sum(v > 0 for v in L)}/{len(L)}')
