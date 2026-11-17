"""Addendum B decision rule: late-awake (stimulus-free gaps in the awake block) vs early awake; induction targets vs late awake.
    python claar_late_sum.py claar_z_ns.npz claar_z_ind.npz claar_z_late.npz"""
import sys, json, numpy as np
from scipy.stats import binom
A, I, L = (np.load(f) for f in sys.argv[1:4]); NULL, DIND, SECOND = .0026, -.0063, 'sub-569073_ses-20210416'
R = lambda Z: Z['ns'][:, 0] - Z['ns'][:, 1:].mean(1)
g = lambda Z, k: np.array([r[k] for r in json.loads(str(Z['rows']))])
rA, rI, rL = R(A), R(I), R(L); sA, stA, sI, sL = g(A, 'session'), g(A, 'state'), g(I, 'session'), g(L, 'session')
biA, dA, biI, dI = g(A, 'burst'), g(A, 'delta_fraction'), g(I, 'burst'), g(I, 'delta_fraction')
def rep(name, d):
    d = np.array(d); n = len(d); neg = int((d < 0).sum()); b = np.random.default_rng(0).choice(d, (10000, n)).mean(1); lo, hi = np.percentile(b, [2.5, 97.5])
    print(f'{name}: n={n}, negative {neg} (p={binom.sf(neg - 1, n, .5):.4f}), mean {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] = {d.mean() / NULL:+.1f} null SD'); return d.mean(), lo
dl, di = [], []
for s in dict.fromkeys(sL):
    if s == SECOND: continue
    a = (sA == s) & (stA == 'awake'); l = sL == s
    if l.sum() < 3: print(f'{s[:10]} late windows {l.sum()} (< 3, dropped)'); continue
    dl.append(rL[l].mean() - rA[a].mean()); t = (sI == s) & (biI >= np.percentile(biA[a], 5)) & (dI >= np.percentile(dA[a], 95))
    print(f'{s[:10]} late windows {l.sum():2d}, Δ_late {dl[-1]:+.4f}' + (f', induction target − late {rI[t].mean() - rL[l].mean():+.4f} ({t.sum()} targets)' if t.sum() >= 3 else ''))
    if t.sum() >= 3: di.append(rI[t].mean() - rL[l].mean())
m, lo = rep('PRIMARY Δ_late (late − early awake)', dl)
print('verdict:', 'DRIFT EXPLAINS' if m <= DIND / 2 else ('DRIFT NEGLIGIBLE' if lo > DIND else 'INCONCLUSIVE'), f'(explains if mean ≤ {DIND / 2:+.4f}; negligible if mean > {DIND / 2:+.4f} and CI lower > {DIND:+.4f})')
if di: rep('secondary induction target − late awake', di)
