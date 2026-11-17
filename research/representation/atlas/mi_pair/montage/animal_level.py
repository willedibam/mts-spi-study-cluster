"""Animal-level inference for the macaque MI pair (normal-scored, real − IAAFT), from zns_[AB]_*.npz: per session Δ = state − awake_ec,
then mean per animal (per agent and pooled over drugs); plus 1−z² real/surrogate increments."""
import glob, numpy as np
from scipy.stats import binom
L = [np.load(f) for f in sorted(glob.glob('zns_[AB]_*.npz'))]; cat = lambda k: np.concatenate([d[k] for d in L]); V = np.concatenate([d['V'] for d in L])
unit, state, agent, animal = cat('unit'), cat('state'), cat('agent'), cat('animal'); res = V[:, 1, 0] - V[:, 1, 1]
D = []
for u in np.unique(unit):
    m = lambda s: (unit == u) & (state == s); s = 'sleep' if agent[unit == u][0] == 'SLP' else 'drug'
    if m(s).any(): D.append((agent[unit == u][0], animal[unit == u][0], res[m(s)].mean() - res[m('awake_ec')].mean(), res[m('awake_eo')].mean() - res[m('awake_ec')].mean()))
ag, an = np.array([d[0] for d in D]), np.array([d[1] for d in D]); dd, eo = np.array([d[2] for d in D]), np.array([d[3] for d in D])
for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP'):
    print(a, {x: round(dd[(ag == a) & (an == x)].mean(), 4) for x in np.unique(an[ag == a])})
dr = ag != 'SLP'; A = {x: dd[dr & (an == x)].mean() for x in np.unique(an[dr])}; E = {x: eo[dr & (an == x)].mean() for x in A}
n = len(A); neg = sum(v < 0 for v in A.values())
print(f'pooled drugs per animal: {dict((k, round(v, 4)) for k, v in A.items())}; {neg}/{n} negative, one-sided sign p={binom.sf(neg - 1, n, .5):.4f}')
print(f'same animals eyes-open − closed: {dict((k, round(v, 4)) for k, v in E.items())}')
# per-animal agent-mean (each agent once per animal), then minimum across animals
print('every animal × agent cell negative:', all(dd[dr & (an == x) & (ag == a)].mean() < 0 for x in A for a in np.unique(ag[dr & (an == x)])),
      f'({sum(1 for x in A for a in np.unique(ag[dr & (an == x)]))} cells)')
d1 = lambda real: 1 - real ** 2
for a in ('KT', 'KTMD', 'MD', 'PF', 'SLP'):
    s = 'sleep' if a == 'SLP' else 'drug'; m = agent == a
    r = [(d1(V[m & (state == t), 1, 0]).mean(), d1(V[m & (state == t), 1, 1]).mean()) for t in ('awake_ec', s)]
    print(f'{a:5s} 1−z² real/surr awake {r[0][0]:.3f}/{r[0][1]:.3f} → {s} {r[1][0]:.3f}/{r[1][1]:.3f}; excess over surrogate {r[0][0]-r[0][1]:+.3f} → {r[1][0]-r[1][1]:+.3f}')
