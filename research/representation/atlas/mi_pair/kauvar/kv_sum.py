"""Frozen decision rule for the Kauvar sEEG ketamine test (kauvar-prereg.md).
    python kv_sum.py kv_z.npz"""
import sys, json, numpy as np
from scipy.stats import binom, spearmanr
Z = np.load(sys.argv[1]); rows = json.loads(str(Z['rows'])); V = Z['asis']; res = V[:, 0] - V[:, 1:].mean(1)
g = lambda k: np.array([r[k] for r in rows]); sub, st, t, bi, dl = g('session'), g('state'), g('start_s'), g('burst'), g('delta_fraction')
D, NL, P, out = [], [], [], []
for s in dict.fromkeys(sub):
    a, i, p = (sub == s) & (st == 'reference'), (sub == s) & (st == 'infusion'), (sub == s) & (st == 'postinfusion')
    if a.sum() < 10 or i.sum() < 10: print(f'{s}: reference {a.sum()}, infusion {i.sum()} windows (< 10, dropped)'); continue
    aw = np.flatnonzero(a); aw = aw[np.argsort(t[aw])]; h = len(aw) // 2
    D.append(res[i].mean() - res[a].mean()); NL.append(res[aw[-h:]].mean() - res[aw[:h]].mean())
    P.append(res[p].mean() - res[a].mean() if p.sum() >= 5 else np.nan)
    print(f'{s}: windows ref/inf/post {a.sum()}/{i.sum()}/{p.sum()}, Δ_inf {D[-1]:+.4f}, Δ_post {P[-1]:+.4f}, split-half {NL[-1]:+.4f} | '
          f'BI {np.median(bi[a]):.2f}/{np.median(bi[i]):.2f} δ {np.median(dl[a]):.2f}/{np.median(dl[i]):.2f} | 1−z² real/surr ref {np.mean(1 - V[a, 0] ** 2):.3f}/{np.mean(1 - V[a, 1:] ** 2):.3f} inf {np.mean(1 - V[i, 0] ** 2):.3f}/{np.mean(1 - V[i, 1:] ** 2):.3f}')
d = np.array(D); n = len(d); sd = np.std(NL, ddof=1); neg = int((d < 0).sum())
b = np.random.default_rng(0).choice(d, (10000, n)).mean(1); lo, hi = np.percentile(b, [2.5, 97.5])
if n < 6: v = 'UNDERPOWERED'
elif neg >= 6 and d.mean() < -2 * sd and hi < 0: v = 'PRESENT'
elif abs(d.mean()) < 2 * sd and lo > -.010: v = 'ABSENT'
else: v = 'INCONCLUSIVE'
print(f'PRIMARY Δ_inf: n={n}, negative {neg} (one-sided sign p={binom.sf(neg - 1, n, .5):.4f}), mean {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], null SD {sd:.4f} → {d.mean() / sd:+.1f} SD → {v}')
pp = np.array(P); pp = pp[np.isfinite(pp)]; print(f'secondary post-infusion − reference: n={len(pp)}, negative {(pp < 0).sum()}, mean {pp.mean():+.4f}')
rho = [spearmanr(bi[(sub == s) & (st == 'infusion')], res[(sub == s) & (st == 'infusion')])[0] for s in dict.fromkeys(sub) if ((sub == s) & (st == 'infusion')).sum() >= 5]
print(f'secondary within-infusion Spearman(residual, BI): median {np.nanmedian(rho):+.2f}, positive {sum(r > 0 for r in rho)}/{len(rho)}')
