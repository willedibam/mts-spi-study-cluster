"""Addendum A decision rule: continuous anaesthetised induction windows (BI ≥ awake p5 and δ ≥ awake p95) vs awake.
    python claar_ind_sum.py claar_z_ns.npz claar_z_ind.npz"""
import sys, json, numpy as np
from scipy.stats import binom, spearmanr
A = np.load(sys.argv[1]); I = np.load(sys.argv[2]); NULL = .0026; SECOND = 'sub-569073_ses-20210416'
ra, ri = json.loads(str(A['rows'])), json.loads(str(I['rows']))
resA = A['ns'][:, 0] - A['ns'][:, 1:].mean(1); resI = I['ns'][:, 0] - I['ns'][:, 1:].mean(1)
g = lambda R, k: np.array([r[k] for r in R])
sA, stA, biA, dA = g(ra, 'session'), g(ra, 'state'), g(ra, 'burst'), g(ra, 'delta_fraction')
sI, biI, dI = g(ri, 'session'), g(ri, 'burst'), g(ri, 'delta_fraction')
out, allind, rho_b, rho_d = [], [], [], []
for s in dict.fromkeys(sI):
    if s == SECOND: continue
    a = (sA == s) & (stA == 'awake'); i = sI == s
    tgt = i & (biI >= np.percentile(biA[a], 5)) & (dI >= np.percentile(dA[a], 95))
    allind.append(resI[i].mean() - resA[a].mean())
    if i.sum() >= 5: rho_b.append(spearmanr(biI[i], resI[i])[0]); rho_d.append(spearmanr(dI[i], resI[i])[0])
    print(f'{s[:10]} induction windows {i.sum():2d}, target {tgt.sum():2d}' + (f', Δ_ind {resI[tgt].mean() - resA[a].mean():+.4f}' if tgt.sum() >= 3 else ''))
    if tgt.sum() >= 3: out.append(resI[tgt].mean() - resA[a].mean())
d = np.array(out); n = len(d)
if n < 8: print(f'UNDERPOWERED: {n} mice with ≥ 3 target windows (need ≥ 8); no verdict')
if n:
    neg = int((d < 0).sum()); k = min(j for j in range(n + 1) if binom.sf(j - 1, n, .5) <= .025)
    b = np.random.default_rng(0).choice(d, (10000, n)).mean(1); lo, hi = np.percentile(b, [2.5, 97.5])
    v = 'PRESENT' if neg >= k and d.mean() < -2 * NULL else ('ABSENT' if abs(d.mean()) < 2 * NULL and lo > -.010 else 'INCONCLUSIVE')
    print(f'PRIMARY Δ_ind: n={n}, negative {neg} (need ≥{k}, p={binom.sf(neg - 1, n, .5):.4f}), mean {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] = {d.mean() / NULL:+.1f} null SD → {v if n >= 8 else "(underpowered) " + v}')
x = np.array(allind); print(f'secondary all induction windows: mean {x.mean():+.4f}, negative {(x < 0).sum()}/{len(x)}')
print(f'secondary within-induction Spearman: residual~BI median {np.median(rho_b):+.2f} ({sum(r > 0 for r in rho_b)}/{len(rho_b)} positive); residual~δ median {np.median(rho_d):+.2f} ({sum(r < 0 for r in rho_d)}/{len(rho_d)} negative)')
