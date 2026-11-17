"""Addendum A checks: BI/speed/rms of target vs awake, stationary-awake reference, BI-slope adjustment, leave-one-out."""
import sys, json, numpy as np
from scipy.stats import binom, linregress
A = np.load('claar_z_ns.npz'); I = np.load('claar_z_ind.npz'); SECOND = 'sub-569073_ses-20210416'
ra, ri = json.loads(str(A['rows'])), json.loads(str(I['rows']))
resA = A['ns'][:, 0] - A['ns'][:, 1:].mean(1); resI = I['ns'][:, 0] - I['ns'][:, 1:].mean(1)
g = lambda R, k: np.array([r[k] for r in R])
sA, stA = g(ra, 'session'), g(ra, 'state'); sI = g(ri, 'session')
f = {k: (g(ra, k), g(ri, k)) for k in ('burst', 'delta_fraction', 'speed', 'rms_uv')}
D, Dst, Dadj, rows = [], [], [], []
for s in dict.fromkeys(sI):
    if s == SECOND: continue
    a = (sA == s) & (stA == 'awake'); i = sI == s
    t = i & (f['burst'][1] >= np.percentile(f['burst'][0][a], 5)) & (f['delta_fraction'][1] >= np.percentile(f['delta_fraction'][0][a], 95))
    if t.sum() < 3: continue
    still = a & (f['speed'][0] <= np.median(f['speed'][0][a]))
    iso = (sA == s) & (stA == 'iso'); sl = linregress(f['burst'][0][iso], resA[iso]).slope   # within-iso residual~BI slope
    d = resI[t].mean() - resA[a].mean(); dbi = f['burst'][1][t].mean() - f['burst'][0][a].mean()
    D.append(d); Dst.append(resI[t].mean() - resA[still].mean()); Dadj.append(d - sl * dbi)
    rows.append((s[:10], t.sum(), round(d, 4), round(dbi, 3), round(sl, 3), round(sl * dbi, 4),
                 round(np.median(f['speed'][1][t]), 1), round(np.median(f['speed'][0][a]), 1),
                 round(np.median(f['rms_uv'][1][t]), 1), round(np.median(f['rms_uv'][0][a]), 1)))
print('session n Δ ΔBI(t−a) isoslope slope·ΔBI speed_t speed_a rms_t rms_a'); [print(*r) for r in rows]
def rep(n_, d):
    d = np.array(d); n = len(d); neg = (d < 0).sum(); b = np.random.default_rng(0).choice(d, (10000, n)).mean(1)
    print(f'{n_:30s} n={n} neg {neg} p={binom.sf(neg-1, n, .5):.4f} mean {d.mean():+.4f} [{np.percentile(b,2.5):+.4f}, {np.percentile(b,97.5):+.4f}] = {d.mean()/.0026:+.1f} SD')
rep('primary', D); rep('vs stationary awake', Dst); rep('BI-slope adjusted (iso slope)', Dadj)
d = np.array(D); print('leave-one-out means', np.round([np.delete(d, j).mean() for j in range(len(d))], 4).tolist())
