"""Post hoc: does the δ filter itself select the drop? High-δ awake windows (early + late, target criteria) vs other awake; residual vs time into induction; late window times."""
import json, numpy as np
from scipy.stats import spearmanr, binom
R = lambda Z: Z['ns'][:, 0] - Z['ns'][:, 1:].mean(1)
Zs = {k: np.load(f'claar_z_{k}.npz') for k in ('ns', 'ind', 'late')}
rows = {k: json.loads(str(Z['rows'])) for k, Z in Zs.items()}; res = {k: R(Z) for k, Z in Zs.items()}
g = lambda k, f: np.array([r[f] for r in rows[k]])
SECOND = 'sub-569073_ses-20210416'; sA, stA = g('ns', 'session'), g('ns', 'state')
aw = stA == 'awake'
# pooled awake = early + late; target criteria use early-awake percentiles
d_hi, n_hi, rho_t, tmin = [], [], [], []
for s in dict.fromkeys(g('late', 'session')):
    if s == SECOND: continue
    a = (sA == s) & aw; l = g('late', 'session') == s
    p5, p95 = np.percentile(g('ns', 'burst')[a], 5), np.percentile(g('ns', 'delta_fraction')[a], 95)
    rr = np.r_[res['ns'][a], res['late'][l]]; dd = np.r_[g('ns', 'delta_fraction')[a], g('late', 'delta_fraction')[l]]; bb = np.r_[g('ns', 'burst')[a], g('late', 'burst')[l]]
    hi = (dd >= p95) & (bb >= p5)
    tmin.append(g('late', 'start_s')[l].min() / 60); tmin.append(g('late', 'start_s')[l].max() / 60)
    if hi.sum() >= 3: d_hi.append(rr[hi].mean() - rr[~hi].mean()); n_hi.append(int(hi.sum()))
print(f'late-awake window start times {min(tmin):.0f}–{max(tmin):.0f} min after recording start')
d = np.array(d_hi); n = len(d); neg = int((d < 0).sum())
print(f'high-δ awake (target criteria) − other awake: n={n} mice (windows {n_hi}), negative {neg} (p={binom.sf(neg-1, n, .5):.3f}), mean {d.mean():+.4f}' if n else 'no mouse has ≥3 high-δ awake windows')
print('per mouse', np.round(d, 4).tolist())
# all awake windows: residual ~ δ within session
r_aw = [spearmanr(g('ns', 'delta_fraction')[(sA == s) & aw], res['ns'][(sA == s) & aw])[0] for s in dict.fromkeys(sA) if s != SECOND]
print(f'within early awake Spearman(residual, δ): median {np.median(r_aw):+.2f}, negative {sum(x < 0 for x in r_aw)}/{len(r_aw)}')
# dose-response: residual vs time into induction
sI = g('ind', 'session'); tI = g('ind', 'start_s')
r_t = [spearmanr(tI[sI == s], res['ind'][sI == s])[0] for s in dict.fromkeys(sI) if s != SECOND]
print(f'within induction Spearman(residual, time): median {np.median(r_t):+.2f}, negative {sum(x < 0 for x in r_t)}/{len(r_t)}')
