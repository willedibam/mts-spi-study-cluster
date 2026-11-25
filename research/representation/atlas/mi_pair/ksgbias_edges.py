"""Post hoc, descriptive (not in the prereg): how the per-edge KSG bias offset varies with G, w and state, real and surrogate.

Offset b = k4 − kshift = the mean k4 of the pair with x_j circularly shifted by T/4, T/2, 3T/4 (bias at independence under the
pair's own serial structure). Also kpool − k4 (the ψ(N) term, ≈ −2w/N). Per window: Spearman ρ over edges; then medians by
agent × state.

  python ksgbias_edges.py <dir with kb_[AB]_*.npz>
"""
import sys, glob, numpy as np
from scipy.stats import spearmanr

L = [np.load(f) for f in sorted(glob.glob(f'{sys.argv[1]}/kb_[AB]_*.npz'))]
X = np.concatenate([q['X'] for q in L]).astype(float); cat = lambda k: np.concatenate([q[k] for q in L])
agent, state = cat('agent'), cat('state')
g, k4, kpool, kshift, w = (X[:, :, c] for c in (0, 1, 2, 4, 5))
b, p = k4 - kshift, kpool - k4
rho = lambda a, c: np.array([[spearmanr(a[i, r], c[i, r])[0] for r in (0, 1)] for i in range(len(X))])  # (W, real/surr)
rbg, rbw, rwg, rpg = rho(b, g), rho(b, w), rho(w, g), rho(p, g)
print('per window: mean over edges, then median over windows. b = k4 − kshift; p = kpool − k4. ρ: Spearman over edges (real | surr)')
print(f'{"agent":5s} {"state":9s} {"n":>4s} {"b real":>7s} {"b surr":>7s} {"real−surr":>9s} {"w real":>6s} '
      f'{"ρ(b,G)":>13s} {"ρ(b,w)":>13s} {"ρ(w,G)":>13s} {"ρ(p,G)":>13s}')
for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP'):
    for s in ('drug', 'sleep', 'awake_ec', 'awake_eo'):
        m = (agent == a) & (state == s)
        if not m.any(): continue
        bm = b[m].mean(2); med = lambda r: f'{np.median(r[m, 0]):+.2f} | {np.median(r[m, 1]):+.2f}'
        print(f'{a:5s} {s:9s} {m.sum():4d} {np.median(bm[:, 0]):+.4f} {np.median(bm[:, 1]):+.4f} {np.median(bm[:, 0] - bm[:, 1]):+9.4f} '
              f'{np.median(w[m, 0].mean(1)):6.1f} {med(rbg):>13s} {med(rbw):>13s} {med(rwg):>13s} {med(rpg):>13s}')
